from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from config import Config
from models.node.base import (
    NodeABC,
    calculate_distance,
    findExecTimeInEachKindOfNode,
)
from models.node.user import UserNode
from models.task import Task


class DeepRLEnvironmentDDPG:
    """
    DDPG environment for soft-task whole-task offloading (delay only).

    Action slots (fixed length):
        [local creator] + [K_fixed closest fixed] + [K_mobile closest mobile] + [cloud?]

    State:
        [dataSize, exec_time,
         remaining_capacity x N,
         est_tx_delay x N]
        Local est_tx_delay = 0 so the Actor can learn the no-transmission advantage.
        No hardcoded "always choose local".

    Reward:
        -(finish_time - release_time)
    """

    EPS = 1e-9
    DEFAULT_TX_BANDWIDTH_BPS = 1e7

    def __init__(self, simulator, xi: float = Config.DDPGNewConfig.BURDEN_XI):
        self.simulator = simulator
        self.metrics = simulator.metrics
        self.xi = float(xi)

        # Use your config names FixedFogNodeCount / MobileFogNodeCount
        self.top_k_fixed = Config.DDPGNewConfig.FixedFogNodeCount
        self.top_k_mobile = Config.DDPGNewConfig.MobileFogNodeCount
        self.tx_bandwidth_bps = Config.DDPGNewConfig.DEFAULT_TX_BANDWIDTH_BPS

        if self.top_k_fixed < 0 or self.top_k_mobile < 0:
            raise ValueError("Top-K fog counts must be >= 0")
        if self.tx_bandwidth_bps <= 0.0:
            raise ValueError("TX_BANDWIDTH_BPS must be > 0")

        self.has_cloud = self.simulator.cloud_node is not None
        self.num_fixed_slots = self.top_k_fixed
        self.num_mobile_slots = self.top_k_mobile

        self.action_dim = (
            1
            + self.num_fixed_slots
            + self.num_mobile_slots
            + (1 if self.has_cloud else 0)
        )
        # 2 task feats + capacity per slot + est_tx per slot
        self.state_dim = 2 + (2 * self.action_dim)

    # ------------------------------------------------------------------
    # Distance / Top-K
    # ------------------------------------------------------------------
    def _closeness_rank_key(self, task: Task, node: NodeABC) -> Tuple:
        """Closer first, then node id."""
        return (
            calculate_distance(task.creator.x, task.creator.y, node.x, node.y),
            str(getattr(node, "id", "")),
        )

    def _select_top_k_nodes(
        self,
        task: Task,
        nodes,
        k: int,
    ) -> List[Optional[NodeABC]]:
        """K closest live nodes; pad with None if fewer than k."""
        live = list(nodes)
        live.sort(key=lambda node: self._closeness_rank_key(task, node))
        selected: List[Optional[NodeABC]] = list(live[:k])
        while len(selected) < k:
            selected.append(None)
        return selected

    def get_candidate_nodes(self, task: Task) -> List[Optional[NodeABC]]:
        """
        Fixed-length action list:
            [local] + K closest fixed + K closest mobile + [cloud?]
        """
        if task.creator is None:
            raise ValueError(
                f"Task {task.id} has no creator; DDPG state cannot be built."
            )

        candidates: List[Optional[NodeABC]] = [task.creator]
        candidates.extend(
            self._select_top_k_nodes(
                task,
                self.simulator.fixed_fog_nodes.values(),
                self.num_fixed_slots,
            )
        )
        candidates.extend(
            self._select_top_k_nodes(
                task,
                self.simulator.mobile_fog_nodes.values(),
                self.num_mobile_slots,
            )
        )
        if self.has_cloud:
            candidates.append(self.simulator.cloud_node)

        if len(candidates) != self.action_dim:
            raise RuntimeError(
                "DDPG candidate roster size mismatch: "
                f"expected {self.action_dim}, got {len(candidates)}."
            )
        return candidates

    # ------------------------------------------------------------------
    # Soft workload
    # ------------------------------------------------------------------
    @staticmethod
    def _is_soft_task(task: Task) -> bool:
        """True if soft (DDPG only uses soft tasks)."""
        return not bool(getattr(task, "is_hard", False))

    @staticmethod
    def _remaining_soft_work(node: Optional[NodeABC]) -> float:
        """Queued soft work left on node."""
        if node is None:
            return 0.0
        total = 0.0
        for queued_task in getattr(node, "tasks", []):
            if bool(getattr(queued_task, "is_hard", False)):
                continue
            total += max(0.0, float(getattr(queued_task, "remaining_time", 0.0)))
        return total

    @staticmethod
    def _effective_soft_capacity(node: Optional[NodeABC]) -> float:
        """Soft capacity (UserNode uses core_us if available)."""
        if node is None:
            return DeepRLEnvironmentDDPG.EPS
        if isinstance(node, UserNode):
            core_us = getattr(node, "core_us", None)
            if core_us is not None and len(core_us) == node.num_cores:
                return max(
                    DeepRLEnvironmentDDPG.EPS,
                    float(sum(max(0.0, float(v)) for v in core_us)),
                )
        return max(DeepRLEnvironmentDDPG.EPS, float(node.num_cores))

    @staticmethod
    def _task_processing_work(task: Task, node: Optional[NodeABC]) -> float:
        """Extra work if whole task is assigned to node."""
        if node is None:
            return float("inf")
        work = float(findExecTimeInEachKindOfNode(task, executor=node))
        if work <= 0.0:
            work = max(
                DeepRLEnvironmentDDPG.EPS,
                float(getattr(task, "exec_time", 0.0)),
            )
        return max(DeepRLEnvironmentDDPG.EPS, work)

    # ------------------------------------------------------------------
    # Est. TX delay (state only)
    # ------------------------------------------------------------------
    def _estimate_tx_delay(self, task: Task, node: Optional[NodeABC]) -> float:
        """Local=0; remote=size/bw + distance/c. Feature only."""
        if node is None:
            return 1e6
        if task.creator is not None and node.id == task.creator.id:
            return 0.0
        data_size = max(0.0, float(getattr(task, "dataSize", 0.0)))
        tx_time = data_size / self.tx_bandwidth_bps
        distance = calculate_distance(task.creator.x, task.creator.y, node.x, node.y)
        prop_time = 0.0 if distance == float("inf") else float(distance) / 3e8
        return float(tx_time + prop_time)

    # ------------------------------------------------------------------
    # Burden
    # ------------------------------------------------------------------
    def get_burden(self, node: Optional[NodeABC]) -> float:
        """remaining_soft_work / soft_capacity."""
        if node is None:
            return float("inf")
        return self._remaining_soft_work(node) / self._effective_soft_capacity(node)

    def calculate_burden_snapshot(
        self,
        task: Task,
    ) -> Tuple[List[Optional[NodeABC]], np.ndarray, float, float]:
        """candidates, burdens, mean burden, xi*mean."""
        candidates = self.get_candidate_nodes(task)
        present = []
        all_b = []
        for node in candidates:
            if node is None:
                all_b.append(float("inf"))
                continue
            b = self.get_burden(node)
            all_b.append(b)
            present.append(b)
        burdens = np.asarray(all_b, dtype=np.float32)
        avg = float(np.mean(present)) if present else 0.0
        return candidates, burdens, avg, self.xi * avg

    def get_prospective_burden(self, task: Task, node: Optional[NodeABC]) -> float:
        """Burden after adding this whole task."""
        if node is None:
            return float("inf")
        work = self._remaining_soft_work(node) + self._task_processing_work(task, node)
        return work / self._effective_soft_capacity(node)

    def can_accept_without_overload(
        self,
        task: Task,
        node: Optional[NodeABC],
        burden_snapshot=None,
    ) -> bool:
        """True if node may take the whole soft task under burden rule."""
        if node is None or not self._is_soft_task(task):
            return False
        if not node.can_offload_task(task):
            return False
        if burden_snapshot is None:
            burden_snapshot = self.calculate_burden_snapshot(task)
        _, _, avg, threshold = burden_snapshot
        if avg <= self.EPS:
            return True
        return self.get_prospective_burden(task, node) <= threshold + self.EPS

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def get_remaining_capacity_vector(self, task: Task) -> np.ndarray:
        """Headroom per slot before xi*avg burden."""
        candidates, _, avg, threshold = self.calculate_burden_snapshot(task)
        out = []
        for node in candidates:
            if node is None:
                out.append(0.0)
                continue
            cap = self._effective_soft_capacity(node)
            cur = self._remaining_soft_work(node)
            if avg <= self.EPS:
                headroom = cap
            else:
                headroom = max(0.0, threshold * cap - cur)
            out.append(float(headroom))
        return np.asarray(out, dtype=np.float32)

    def get_est_tx_delay_vector(self, task: Task) -> np.ndarray:
        """Per-slot est tx delay (local=0)."""
        candidates = self.get_candidate_nodes(task)
        return np.asarray(
            [self._estimate_tx_delay(task, n) for n in candidates],
            dtype=np.float32,
        )

    def _get_state(
        self,
        task: Optional[Task] = None,
        current_time: Optional[float] = None,
    ) -> np.ndarray:
        """
        [dataSize, exec_time, remaining_capacity x N, est_tx_delay x N]
        """
        if task is None:
            return np.zeros(self.state_dim, dtype=np.float32)
        if not self._is_soft_task(task):
            raise ValueError(f"Hard task {task.id} must not enter DDPG env.")

        task_features = np.asarray(
            [
                max(0.0, float(getattr(task, "dataSize", 0.0))),
                max(0.0, float(getattr(task, "exec_time", 0.0))),
            ],
            dtype=np.float32,
        )
        state = np.concatenate(
            [
                task_features,
                self.get_remaining_capacity_vector(task),
                self.get_est_tx_delay_vector(task),
            ]
        ).astype(np.float32)

        if state.shape[0] != self.state_dim:
            raise RuntimeError(
                f"DDPG state dim mismatch: expected {self.state_dim}, got {state.shape[0]}."
            )
        return state

    # ------------------------------------------------------------------
    # Mask
    # ------------------------------------------------------------------
    @staticmethod
    def _physical_feasibility_mask(
        task: Task,
        candidates: List[Optional[NodeABC]],
    ) -> np.ndarray:
        """1 if node can_offload_task."""
        mask = np.zeros(len(candidates), dtype=np.float32)
        for i, node in enumerate(candidates):
            if node is not None and node.can_offload_task(task):
                mask[i] = 1.0
        return mask

    def get_action_mask(self, task: Task) -> np.ndarray:
        """
        Feasible + burden OK; fallback currently underloaded.
        None slots masked. No forced local.
        """
        if not self._is_soft_task(task):
            return np.zeros(self.action_dim, dtype=np.float32)

        candidates, burdens, avg, threshold = self.calculate_burden_snapshot(task)
        physical = self._physical_feasibility_mask(task, candidates)

        if avg <= self.EPS:
            return physical

        prospective = np.zeros(len(candidates), dtype=np.float32)
        for i, node in enumerate(candidates):
            if physical[i] <= 0.0 or node is None:
                continue
            if self.get_prospective_burden(task, node) <= threshold + self.EPS:
                prospective[i] = 1.0
        if np.any(prospective > 0.0):
            return prospective

        under = np.zeros(len(candidates), dtype=np.float32)
        for i, node in enumerate(candidates):
            if physical[i] <= 0.0 or node is None:
                continue
            if float(burdens[i]) <= threshold + self.EPS:
                under[i] = 1.0
        return under

    def is_candidate_allowed(self, task: Task, node: Optional[NodeABC]) -> bool:
        """Commit-time recheck of one node against current mask."""
        if node is None:
            return False
        candidates = self.get_candidate_nodes(task)
        mask = self.get_action_mask(task)
        for i, c in enumerate(candidates):
            if c is not None and c.id == node.id:
                return bool(mask[i] > 0.0)
        return False

    # ------------------------------------------------------------------
    # Index mapping
    # ------------------------------------------------------------------
    def get_executor_from_action_index(
        self,
        task: Task,
        action_index: int,
    ) -> Optional[NodeABC]:
        """Slot index -> node."""
        candidates = self.get_candidate_nodes(task)
        action_index = int(action_index)
        if 0 <= action_index < len(candidates):
            return candidates[action_index]
        return None

    def get_action_from_executor(self, task: Task, executor: NodeABC) -> int:
        """Node -> slot index for this task."""
        for i, node in enumerate(self.get_candidate_nodes(task)):
            if node is not None and node.id == executor.id:
                return i
        raise ValueError(f"Executor {executor.id} is not in the DDPG candidate set.")

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_reward(
        self,
        task: Task,
        executor: Optional[NodeABC] = None,
        all_fog_nodes=None,
    ) -> float:
        """reward = -(finish_time - release_time)."""
        if not self._is_soft_task(task):
            raise ValueError(f"Hard task {task.id} must not produce a DDPG reward.")
        if float(task.finish_time) <= 0.0:
            raise ValueError(f"Task {task.id} has not finished; no delay reward.")
        delay = max(0.0, float(task.finish_time) - float(task.release_time))
        return -delay
