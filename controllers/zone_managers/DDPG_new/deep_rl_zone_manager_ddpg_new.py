from __future__ import annotations

from typing import Any, Iterable, Optional, Unpack

import numpy as np

from config import Config
from controllers.zone_managers.base import ZoneManagerABC, ZoneManagerUpdate

from .ddpg_agent import DDPGAgent
from .deep_rl_env_ddpg import DeepRLEnvironmentDDPG

from models.node.base import NodeABC
from models.task import Task


class DeepRLZoneManager_DDPG_New(ZoneManagerABC):
    """
    ZoneManager for the final adapted Xu et al. DDPG baseline.

    Responsibilities:
      1) Handle SOFT tasks only.
      2) Share one simulator-wide DDPG Actor/Critic across all zones.
      3) Build/preserve the exact Actor state.
      4) Ask Actor for one continuous score per candidate executor.
      5) Apply the Environment feasibility/burden mask.
      6) Select exactly one whole-task executor from Actor ranking.
      7) Preserve state/action for SimulatorDDPGnew and replay memory.

    This class does NOT execute tasks, simulate communication, calculate
    rewards, split tasks, or use a post-Actor Greedy optimizer.
    """

    def __init__(self, zone):
        super().__init__(zone)

        self.simulator = None
        self.env: Optional[DeepRLEnvironmentDDPG] = None
        self.agent: Optional[DDPGAgent] = None

        self.__target_node: Optional[NodeABC] = None

    @staticmethod
    def _cfg(name: str, default):
        cfg = getattr(Config, "DDPGNewConfig", None)
        if cfg is None:
            return default
        return getattr(cfg, name, default)

    def set_simulator(self, simulator):
        """
        All DDPG ZoneManagers share one simulator-wide environment and agent.
        """
        self.simulator = simulator

        if not hasattr(simulator, "_ddpg_baseline_env"):
            simulator._ddpg_baseline_env = DeepRLEnvironmentDDPG(
                simulator,
                xi=float(self._cfg("BURDEN_XI", 1.5)),
            )

        self.env = simulator._ddpg_baseline_env

        if not hasattr(simulator, "_ddpg_baseline_agent"):
            simulator._ddpg_baseline_agent = DDPGAgent(
                state_dim=self.env.state_dim,
                action_dim=self.env.action_dim,
                max_action=float(self._cfg("MAX_ACTION", 1.0)),
                actor_lr=float(self._cfg("ACTOR_LR", 0.001)),
                critic_lr=float(self._cfg("CRITIC_LR", 0.002)),
                gamma=float(self._cfg("GAMMA", 0.99)),
                tau=float(self._cfg("TAU", 0.01)),
                memory_size=int(self._cfg("REPLAY_BUFFER_SIZE", 10000)),
            )

        self.agent = simulator._ddpg_baseline_agent

    def _creator_is_in_this_zone(self, task: Task) -> bool:
        if task.creator is None:
            return False

        return self.zone.is_in_coverage(
            task.creator.x,
            task.creator.y,
        )

    def can_offload_task(self, task: Task) -> bool:
        """
        Only SOFT tasks are eligible for DDPG.
        """
        if bool(getattr(task, "is_hard", False)):
            return False

        if task.creator is None:
            return False

        if self.env is None or self.agent is None:
            return False

        if not self._creator_is_in_this_zone(task):
            return False

        action_mask = self.env.get_action_mask(task)

        return bool(
            np.any(
                np.asarray(action_mask, dtype=np.float32) > 0.0
            )
        )

    def get_ranked_action_indices(
        self,
        task: Task,
        continuous_action,
        action_mask=None,
        excluded_node_ids: Optional[Iterable[Any]] = None,
    ) -> list[int]:
        """
        Return valid candidate indices sorted by Actor score.

        SimulatorDDPGnew can reuse this ranking at commit time if the first
        choice becomes invalid due to concurrent load changes.
        """
        if self.env is None:
            raise RuntimeError("DDPG ZoneManager has no environment.")

        scores = np.asarray(
            continuous_action,
            dtype=np.float32,
        ).reshape(-1)

        if scores.size != self.env.action_dim:
            raise ValueError(
                "DDPG action dimension mismatch: "
                f"received {scores.size}, expected {self.env.action_dim}."
            )

        if action_mask is None:
            action_mask = self.env.get_action_mask(task)

        mask = np.asarray(
            action_mask,
            dtype=np.float32,
        ).reshape(-1)

        if mask.size != scores.size:
            raise ValueError(
                "DDPG action/mask dimension mismatch: "
                f"{scores.size} vs {mask.size}."
            )

        excluded = {
            str(node_id)
            for node_id in (excluded_node_ids or [])
        }

        candidates = self.env.get_candidate_nodes(task)

        valid_indices = []

        for idx, node in enumerate(candidates):
            if node is None:
                continue

            if mask[idx] <= 0.0:
                continue

            if str(node.id) in excluded:
                continue

            valid_indices.append(idx)

        valid_indices.sort(
            key=lambda idx: float(scores[idx]),
            reverse=True,
        )

        return valid_indices

    def select_executor_from_action(
        self,
        task: Task,
        continuous_action,
        action_mask=None,
        excluded_node_ids: Optional[Iterable[Any]] = None,
    ) -> tuple[Optional[NodeABC], Optional[int]]:
        """
        Map Actor scores to one valid whole-task executor.
        """
        ranked_indices = self.get_ranked_action_indices(
            task,
            continuous_action,
            action_mask=action_mask,
            excluded_node_ids=excluded_node_ids,
        )

        if not ranked_indices:
            return None, None

        action_index = ranked_indices[0]

        executor = self.env.get_executor_from_action_index(
            task,
            action_index,
        )

        return executor, action_index

    def _get_decision_cache(self) -> dict:
        """
        Prevent multiple exploration-noise samples for one task/time when
        overlapping ZoneManagers are queried.
        """
        if self.simulator is None:
            raise RuntimeError(
                "DDPG ZoneManager is not connected to a simulator."
            )

        if not hasattr(self.simulator, "_ddpg_decision_cache"):
            self.simulator._ddpg_decision_cache = {}

        return self.simulator._ddpg_decision_cache

    def propose_candidate(
        self,
        task: Task,
        current_time: float,
    ):
        """
        Return:
            (
                zone_manager,
                proposed_executor,
                continuous_action_vector,
                exact_actor_state,
            )

        No partial offloading and no Greedy post-processing are used.
        """
        if bool(getattr(task, "is_hard", False)):
            raise ValueError(
                f"Hard task {task.id} must not enter DDPG."
            )

        if self.env is None or self.agent is None:
            raise RuntimeError(
                "DDPG ZoneManager has not been connected to a simulator."
            )

        decision_cache = self._get_decision_cache()
        cached = decision_cache.get(task.id)

        if (
            cached is not None
            and float(cached["current_time"]) == float(current_time)
        ):
            self.__target_node = cached["executor"]

            return (
                cached["zone_manager"],
                cached["executor"],
                cached["continuous_action"].copy(),
                cached["state"].copy(),
            )

        state = self.env._get_state(
            task=task,
            current_time=current_time,
        )

        action_mask = self.env.get_action_mask(task)

        continuous_action = self.agent.select_action(
            state,
            exploration_noise=float(
                self._cfg("EXPLORATION_NOISE", 0.1)
            ),
        )

        executor, action_index = self.select_executor_from_action(
            task,
            continuous_action,
            action_mask=action_mask,
        )

        self.__target_node = executor

        task.ddpg_actor_action_index = action_index
        task.ddpg_actor_executor_id = (
            executor.id if executor is not None else None
        )
        task.ddpg_actor_scores = np.asarray(
            continuous_action,
            dtype=np.float32,
        ).copy()

        decision_cache[task.id] = {
            "current_time": float(current_time),
            "zone_manager": self,
            "executor": executor,
            "continuous_action": np.asarray(
                continuous_action,
                dtype=np.float32,
            ).copy(),
            "state": np.asarray(
                state,
                dtype=np.float32,
            ).copy(),
        }

        return (
            self,
            executor,
            continuous_action,
            state,
        )

    def assign_task(self, task: Task) -> Any:
        """
        Compatibility with ZoneManagerABC. Normal DDPG execution uses
        propose_candidate() so the exact Actor state/action are preserved.
        """
        if bool(getattr(task, "is_hard", False)):
            return None

        if self.__target_node is not None:
            return self.__target_node

        if self.simulator is None:
            raise RuntimeError(
                "DDPG ZoneManager is not connected to a simulator."
            )

        current_time = self.simulator.clock.get_current_time()

        _, executor, _, _ = self.propose_candidate(
            task,
            current_time,
        )

        return executor

    def update(
        self,
        **kwargs: Unpack[ZoneManagerUpdate],
    ):
        """
        SimulatorDDPGnew owns transition completion and training.
        """
        pass