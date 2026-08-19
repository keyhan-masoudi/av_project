from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import numpy as np

from config import Config
from controllers.simulator import Simulator, calcAttenuation
from controllers.zone_managers.DDPG_new.deep_rl_zone_manager_ddpg_new import (
    DeepRLZoneManager_DDPG,
)
from models.node.cloud import CloudNode
from models.node.fog import FixedFogNode, MobileFogNode
from NoiseConfigs.utilsFunctions import UtilsFunc


class SimulatorDDPGnew(Simulator):
    """
    Final simulator adapter for the adapted Xu et al. DDPG baseline.

    Architecture
    ------------
    Hard tasks:
        They are handled by the project's normal hard-task path and remain
        local on their creator. They never participate in DDPG.

    Soft tasks:
        Every soft task reaching the offloading path is handled by DDPG.
        The Actor selects exactly one executor from:
            creator + fixed fogs + mobile fogs + cloud.

    The simulator remains responsible for:
        - actual communication/noise/packet loss;
        - retransmission;
        - final Node.assign_task(...);
        - execution;
        - metrics;
        - delayed delay-only reward;
        - replay-transition completion.

    There is no partial offloading and no energy objective.
    """

    # ==================================================================
    # Config helper
    # ==================================================================
    @staticmethod
    def _cfg(name: str, default):
        cfg = getattr(Config, "DDPGNewConfig", None)
        if cfg is None:
            return default
        return getattr(cfg, name, default)

    # ==================================================================
    # DDPG transition bookkeeping
    # ==================================================================
    def _get_ddpg_transition_records(self) -> Dict[Any, dict]:
        """
        One delayed transition record per successfully assigned soft task.

        A record can receive s_{t+1} before its task finishes, which is
        necessary because many soft tasks execute concurrently.
        """
        if not hasattr(self, "_ddpg_transition_records"):
            self._ddpg_transition_records = {}

        return self._ddpg_transition_records

    def _get_last_ddpg_decision_task_id(self):
        return getattr(
            self,
            "_ddpg_last_decision_task_id",
            None,
        )

    def _set_last_ddpg_decision_task_id(self, task_id) -> None:
        self._ddpg_last_decision_task_id = task_id

    def _train_agent_once(self, agent) -> None:
        agent.train(
            batch_size=int(
                self._cfg(
                    "BATCH_SIZE",
                    256,
                )
            )
        )

    def _try_store_ddpg_transition(self, task_id) -> bool:
        """
        Store a transition only when both delayed pieces are available:

            reward       -> known when the soft task finishes
            next_state   -> known at the next successful DDPG decision

        This fixes the old concurrent-task problem where a task that finished
        late was incorrectly connected to the first decision AFTER completion.
        """
        records = self._get_ddpg_transition_records()
        record = records.get(task_id)

        if record is None:
            return False

        if record["reward"] is None:
            return False

        if record["next_state"] is None:
            return False

        agent = record["agent"]

        agent.store_experience(
            record["state"],
            record["action"],
            record["reward"],
            record["next_state"],
            done=record["done"],
        )

        del records[task_id]

        self._train_agent_once(agent)

        return True

    def _register_successful_ddpg_decision(
        self,
        *,
        task,
        zone_manager: DeepRLZoneManager_DDPG,
        actor_state: np.ndarray,
        continuous_action: np.ndarray,
        current_time: float,
    ) -> None:
        """
        Register a successful whole-task assignment as one DDPG decision.

        Before creating the current record, the exact current Actor state is
        attached as s_{t+1} to the PREVIOUS successful DDPG decision.

        This happens at decision time, not at task-completion time, so
        overlapping/concurrent soft tasks produce the correct temporal chain:

            decision A -> decision B -> decision C

        even if completion order is different.
        """
        records = self._get_ddpg_transition_records()

        actor_state = np.asarray(
            actor_state,
            dtype=np.float32,
        ).copy()

        continuous_action = np.asarray(
            continuous_action,
            dtype=np.float32,
        ).copy()

        previous_task_id = (
            self._get_last_ddpg_decision_task_id()
        )

        if (
            previous_task_id is not None
            and previous_task_id in records
            and records[previous_task_id]["next_state"] is None
        ):
            records[previous_task_id]["next_state"] = (
                actor_state.copy()
            )
            records[previous_task_id]["done"] = False

            # If the previous task already finished, the transition can now
            # immediately enter replay memory.
            self._try_store_ddpg_transition(
                previous_task_id
            )

        # A successfully assigned task should have exactly one active RL record.
        records[task.id] = {
            "agent": zone_manager.agent,
            "state": actor_state,
            "action": continuous_action,
            "reward": None,
            "next_state": None,
            "done": False,
            "decision_time": float(current_time),
        }

        self._set_last_ddpg_decision_task_id(
            task.id
        )

    def _complete_ddpg_reward(self, task) -> None:
        """
        Fill the delayed delay-only reward when the soft task finishes.

        Reward:
            -(finish_time - release_time)
        """
        records = self._get_ddpg_transition_records()
        record = records.get(task.id)

        if record is None:
            return

        rl_zm = getattr(
            task,
            "rl_zone_manager",
            None,
        )

        if (
            rl_zm is None
            or not isinstance(
                rl_zm,
                DeepRLZoneManager_DDPG,
            )
        ):
            return

        reward = rl_zm.env._compute_reward(
            task,
            task.executor,
            None,
        )

        record["reward"] = float(reward)

        self.metrics.add_reward(
            float(reward)
        )

        # If the next decision already happened, this transition is complete.
        self._try_store_ddpg_transition(
            task.id
        )

    def finalize_ddpg_episode(self) -> int:
        """
        Finalize only transitions whose real completion reward is known.

        If a completed task is the final DDPG decision, it has no later
        s_{t+1}; use a terminal zero state with done=True.

        Tasks that never completed have no valid delay-only reward and are
        therefore discarded rather than receiving an invented penalty.
        """
        records = self._get_ddpg_transition_records()

        if not records:
            return 0

        stored = 0

        for task_id in list(records.keys()):
            record = records.get(task_id)

            if record is None:
                continue

            # No measured finish time => no legitimate delay-only reward.
            if record["reward"] is None:
                del records[task_id]
                continue

            if record["next_state"] is None:
                record["next_state"] = np.zeros(
                    record["agent"].state_dim,
                    dtype=np.float32,
                )
                record["done"] = True

            if self._try_store_ddpg_transition(
                task_id
            ):
                stored += 1

        self._ddpg_last_decision_task_id = None

        return stored

    # ==================================================================
    # Decision-cache cleanup
    # ==================================================================
    def _clear_ddpg_decision_cache(self, task) -> None:
        cache = getattr(
            self,
            "_ddpg_decision_cache",
            None,
        )

        if cache is not None:
            cache.pop(
                task.id,
                None,
            )

    # ==================================================================
    # ZoneManager proposal collection
    # ==================================================================
    def find_zone_manager_offload_task(
        self,
        zone_managers,
        task,
        current_time,
    ):
        """
        Obtain ONE global DDPG proposal for a soft task.

        All DDPG ZoneManagers share the same Actor and global candidate set,
        therefore overlapping zones must not create competing Actor decisions.

        Return format:
            (
                zone_manager,
                proposed_executor,
                continuous_action_vector,
                exact_actor_state,
            )
        """
        if bool(
            getattr(
                task,
                "is_hard",
                False,
            )
        ):
            # Hard tasks belong to the project's local hard-task path.
            return []

        for zone_manager in zone_managers:
            if not isinstance(
                zone_manager,
                DeepRLZoneManager_DDPG,
            ):
                continue

            if not zone_manager.can_offload_task(
                task
            ):
                continue

            proposal = zone_manager.propose_candidate(
                task,
                current_time,
            )

            (
                proposed_zone_manager,
                proposed_executor,
                continuous_action,
                actor_state,
            ) = proposal

            if (
                proposed_executor is None
                or actor_state is None
                or continuous_action is None
            ):
                continue

            # One DDPG decision is enough because the policy/candidate set is
            # simulator-wide.
            return [
                (
                    proposed_zone_manager,
                    proposed_executor,
                    continuous_action,
                    actor_state,
                )
            ]

        return []

    # ==================================================================
    # Actor-ranked commit-time validation
    # ==================================================================
    def _resolve_current_actor_choice(
        self,
        *,
        zone_manager: DeepRLZoneManager_DDPG,
        task,
        continuous_action,
    ):
        """
        Rebuild the CURRENT mask immediately before communication/assignment.

        If the Actor's original top choice became invalid because earlier soft
        tasks were assigned in the meantime, select the next-highest valid
        Actor-ranked node.

        No Greedy delay heuristic is used.
        """
        current_mask = (
            zone_manager.env.get_action_mask(
                task
            )
        )

        executor, action_index = (
            zone_manager.select_executor_from_action(
                task,
                continuous_action,
                action_mask=current_mask,
            )
        )

        return executor, action_index

    # ==================================================================
    # Physical assignment
    # ==================================================================
    def choose_executor_and_assign(
        self,
        zone_manager_offload_task,
        task,
        partitions,
        current_time,
    ):
        """
        Commit the DDPG whole-task action through the real simulator.

        Important behavior:
          - no forced local/cloud fallback when no DDPG action is admissible;
          - no artificial NEGATIVE_REWARD;
          - local execution bypasses network packet loss;
          - remote execution keeps the project's attenuation/noise/PLR model;
          - exactly one executor receives the whole task.
        """
        if bool(
            getattr(
                task,
                "is_hard",
                False,
            )
        ):
            # Hard tasks should never reach this method.
            return

        if len(zone_manager_offload_task) == 0:
            # Do not bypass the burden/admission mechanism by forcing the task
            # locally or to cloud. Retry later; deadline misses remain natural
            # consequences of insufficient capacity.
            self._clear_ddpg_decision_cache(
                task
            )

            self.schedule_retransmission(
                task,
                current_time
                + Config.SimulatorConfig.TIMEOUT_TIME,
            )
            return

        (
            chosen_zone_manager,
            proposed_executor,
            continuous_action,
            actor_state,
        ) = zone_manager_offload_task[0]

        if not isinstance(
            chosen_zone_manager,
            DeepRLZoneManager_DDPG,
        ):
            raise TypeError(
                "SimulatorDDPGnew received a non-DDPG ZoneManager."
            )

        # --------------------------------------------------------------
        # Commit-time recheck using the SAME Actor score vector.
        # --------------------------------------------------------------
        chosen_executor, action_index = (
            self._resolve_current_actor_choice(
                zone_manager=chosen_zone_manager,
                task=task,
                continuous_action=continuous_action,
            )
        )

        if chosen_executor is None:
            self._clear_ddpg_decision_cache(
                task
            )

            self.schedule_retransmission(
                task,
                current_time
                + Config.SimulatorConfig.TIMEOUT_TIME,
            )
            return

        # Diagnostics: initial proposal vs final Actor-ranked commit choice.
        task.ddpg_proposed_executor_id = (
            proposed_executor.id
            if proposed_executor is not None
            else None
        )
        task.ddpg_final_executor_id = (
            chosen_executor.id
        )
        task.ddpg_final_action_index = (
            action_index
        )

        # --------------------------------------------------------------
        # LOCAL SOFT EXECUTION
        # --------------------------------------------------------------
        if (
            chosen_executor.id
            == task.creator.id
        ):
            # There is no wireless transmission for local execution.
            self.task_zone_managers[
                task.id
            ] = chosen_zone_manager

            task.rl_state = np.asarray(
                actor_state,
                dtype=np.float32,
            ).copy()

            task.rl_continuous_action = (
                np.asarray(
                    continuous_action,
                    dtype=np.float32,
                ).copy()
            )

            task.rl_action = int(
                action_index
            )
            task.rl_zone_manager = (
                chosen_zone_manager
            )

            self._register_successful_ddpg_decision(
                task=task,
                zone_manager=chosen_zone_manager,
                actor_state=actor_state,
                continuous_action=continuous_action,
                current_time=current_time,
            )

            chosen_executor.assign_task(
                task,
                current_time,
                self.fixed_fog_nodes,
            )

            self._clear_ddpg_decision_cache(
                task
            )
            return

        # --------------------------------------------------------------
        # REMOTE SOFT EXECUTION: preserve project network model
        # --------------------------------------------------------------
        intersecting_partitions = (
            UtilsFunc().find_line_intersections(
                (
                    task.creator.x,
                    task.creator.y,
                ),
                (
                    chosen_executor.x,
                    chosen_executor.y,
                ),
                partitions,
            )
        )

        if isinstance(
            chosen_executor,
            CloudNode,
        ):
            attenuation = self.calcAttForCloud(
                task,
                intersecting_partitions,
            )

        elif len(
            intersecting_partitions
        ) > 0:
            attenuation = calcAttenuation(
                task,
                chosen_executor,
                intersecting_partitions,
            )

        else:
            attenuation = 0

        attenuation_list = [
            (
                chosen_zone_manager,
                chosen_executor,
                attenuation,
                continuous_action,
            )
        ]

        final_choice, plr = (
            self.noise_controller.makeFinalChoice(
                attenuation_list,
                task,
                partitions,
                Config.NoiseMethod.DEFAULT_METHOD,
            )
        )

        if not final_choice:
            self.metrics.inc_no_device_found_to_run_becauseOf_Noise()

            self._clear_ddpg_decision_cache(
                task
            )

            self.schedule_retransmission(
                task,
                current_time
                + Config.SimulatorConfig.TIMEOUT_TIME,
            )
            return

        (
            final_zone_manager,
            final_executor,
            _,
            final_continuous_action,
        ) = final_choice

        # With one Actor-selected candidate the noise controller must not
        # silently change the RL destination.
        if (
            final_executor is None
            or final_executor.id
            != chosen_executor.id
        ):
            self._clear_ddpg_decision_cache(
                task
            )

            self.schedule_retransmission(
                task,
                current_time
                + Config.SimulatorConfig.TIMEOUT_TIME,
            )
            return

        packet_loss_random_number = (
            random.randint(
                0,
                100,
            )
        )

        if (
            packet_loss_random_number
            < plr
        ):
            self.metrics.inc_packet_loss()

            self._clear_ddpg_decision_cache(
                task
            )

            self.schedule_retransmission(
                task,
                current_time
                + Config.SimulatorConfig.TIMEOUT_TIME,
            )
            return

        # --------------------------------------------------------------
        # Successful remote assignment
        # --------------------------------------------------------------
        self.task_zone_managers[
            task.id
        ] = final_zone_manager

        task.rl_state = np.asarray(
            actor_state,
            dtype=np.float32,
        ).copy()

        task.rl_continuous_action = (
            np.asarray(
                final_continuous_action,
                dtype=np.float32,
            ).copy()
        )

        task.rl_action = int(
            action_index
        )
        task.rl_zone_manager = (
            final_zone_manager
        )

        self._register_successful_ddpg_decision(
            task=task,
            zone_manager=final_zone_manager,
            actor_state=actor_state,
            continuous_action=final_continuous_action,
            current_time=current_time,
        )

        final_executor.assign_task(
            task,
            current_time,
            self.fixed_fog_nodes,
        )

        self._clear_ddpg_decision_cache(
            task
        )

    # ==================================================================
    # Execution + metrics + delayed reward
    # ==================================================================
    def execute_tasks_for_one_step(self):
        """
        Execute one simulator tick and attach the real delay-only reward to
        completed DDPG soft tasks.

        Hard-task and common executor/deadline metrics are kept compatible
        with the existing simulator architecture.
        """
        executed_tasks = []

        merged_nodes = {
            **self.mobile_fog_nodes,
            **self.user_nodes,
            **self.fixed_fog_nodes,
        }

        if self.cloud_node is not None:
            merged_nodes[
                self.cloud_node.id
            ] = self.cloud_node

        for _, node in merged_nodes.items():
            tasks = node.execute_tasks(
                self.clock.get_current_time(),
                self.fixed_fog_nodes,
            )

            executed_tasks.extend(
                tasks
            )

            for task in tasks:
                # ------------------------------------------------------
                # Existing load-difference metric
                # ------------------------------------------------------
                zone_manager = (
                    self.task_zone_managers.get(
                        task.id
                    )
                )

                if zone_manager:
                    all_fog_nodes = {
                        **zone_manager.fixed_fog_nodes,
                        **zone_manager.mobile_fog_nodes,
                    }

                    loads = [
                        len(fog.tasks)
                        for fog
                        in all_fog_nodes.values()
                        if fog.can_offload_task(task)
                    ]

                    if loads:
                        self.metrics.inc_task_load_diff(
                            task.id,
                            min(loads),
                            max(loads),
                        )

                # ------------------------------------------------------
                # Executor metrics
                # ------------------------------------------------------
                if bool(
                    getattr(
                        task,
                        "is_hard",
                        False,
                    )
                ):
                    self.metrics.inc_local_hard_execution()

                else:
                    if isinstance(
                        task.executor,
                        (
                            FixedFogNode,
                            MobileFogNode,
                        ),
                    ):
                        self.metrics.inc_fog_execution()

                    elif (
                        task.creator.id
                        == task.executor.id
                    ):
                        self.metrics.inc_local_execution()

                    elif isinstance(
                        task.executor,
                        CloudNode,
                    ):
                        self.metrics.inc_cloud_tasks()

                    # --------------------------------------------------
                    # Real DDPG reward only after full soft completion.
                    # --------------------------------------------------
                    if (
                        getattr(
                            task,
                            "rl_zone_manager",
                            None,
                        )
                        is not None
                    ):
                        self._complete_ddpg_reward(
                            task
                        )

                # ------------------------------------------------------
                # Deadline metrics remain evaluation-only.
                # ------------------------------------------------------
                if task.is_deadline_missed:
                    missed_info = {
                        "task_id": task.id,
                        "release_time": task.release_time,
                        "deadline": task.deadline,
                        "exec_time": task.exec_time,
                        "finish_time": task.finish_time,
                        "executor_id": task.executor.id,
                        "data_size": task.dataSize,
                        "deadline_diff": (
                            task.finish_time
                            - task.deadline
                        ),
                    }

                    self.missed_deadline_data.append(
                        missed_info
                    )

                    if task.is_hard:
                        self.metrics.inc_hard_deadline_miss()
                    else:
                        self.metrics.inc_deadline_miss()

                else:
                    success_task_info = {
                        "task_id": task.id,
                        "release_time": task.release_time,
                        "deadline": task.deadline,
                        "exec_time": task.exec_time,
                        "finish_time": task.finish_time,
                        "executor_id": task.executor.id,
                        "data_size": task.dataSize,
                        "deadline_diff": (
                            task.finish_time
                            - task.deadline
                        ),
                    }

                    self.success_deadline_data.append(
                        success_task_info
                    )

                    self.metrics.inc_completed_task()

        # Node.execute_tasks() processes one simulator-time unit per call.
        # start_simulation() normally runs while current_time < duration, so
        # finalize on the final executable tick rather than waiting for a time
        # value that may never enter this method.
        if (
            self.clock.get_current_time() + 1.0
            >= Config.SimulatorConfig.SIMULATION_DURATION
        ):
            self.finalize_ddpg_episode()

        return executed_tasks