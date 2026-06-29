import random
from typing import List, Dict

from config import Config
from controllers.zone_managers.PPO.deep_rl_zone_manager_PPO import DeepRLZoneManagerPPO
from models.node.cloud import CloudNode
from NoiseConfigs.utilsFunctions import UtilsFunc
from models.node.fog import FixedFogNode, MobileFogNode
from models.node.base import NodeABC
from models.task import Task
from controllers.simulator import Simulator, calcAttenuation, yellow_bg

class SimulatorPPO(Simulator):

    def __init__(self, loader, clock, cloud):
        super().__init__(loader, clock, cloud)

        self.update_timestep = 1000
        self.time_step_counter = 0

    def find_zone_manager_offload_task(self, zone_managers, task, current_time):
        """
        PPO returns (zm, exec, ppo_data)
        """
        # print("888888888888888888888888888888888888888888")
        zone_manager_offload_task = []
        for zone_manager in zone_managers:
            if zone_manager.can_offload_task(task):
                if hasattr(zone_manager, "propose_candidate"):
                    proposed_zone_manager, proposed_executor, ppo_data = zone_manager.propose_candidate(task,
                                                                                                        current_time)
                    candidate_tuple = (proposed_zone_manager, proposed_executor, ppo_data)
                else:
                    proposed_executor = zone_manager.offload_task(task, current_time)
                    candidate_tuple = (zone_manager, proposed_executor, None)

                if proposed_executor and candidate_tuple not in zone_manager_offload_task:
                    zone_manager_offload_task.append(candidate_tuple)
        return zone_manager_offload_task

    def choose_executor_and_assign(self, zone_manager_offload_task, task, partitions, current_time):
        # print("+++++++++++++++++++++++++++++++++++++++++")
        if len(zone_manager_offload_task) != 0:
            attenuationList = []

            for candidate in zone_manager_offload_task:
                zone_manager, candidate_executor, ppo_data = candidate
                intersecting_partitions = UtilsFunc().find_line_intersections(
                    (task.creator.x, task.creator.y),
                    (candidate_executor.x, candidate_executor.y),
                    partitions
                )

                if len(intersecting_partitions) > 0 and not isinstance(candidate_executor, CloudNode):
                    attenuation = calcAttenuation(task, candidate_executor, intersecting_partitions)
                elif isinstance(candidate_executor, CloudNode):
                    attenuation = self.calcAttForCloud(task, intersecting_partitions)
                else:
                    attenuation = 0

                attenuationList.append((zone_manager, candidate_executor, attenuation, ppo_data))

            finalChoiceToOffload, plr = self.noise_controller.makeFinalChoice(
                attenuationList, task, partitions, Config.NoiseMethod.DEFAULT_METHOD
            )
            # if plr is not None:
            #     # print(f"test:{plr}")
            #     if 0 <= plr < 20:
            #         self.plr_distribution["0-20"] += 1
            #     elif 20 <= plr < 40:
            #         self.plr_distribution["20-40"] += 1
            #     elif 40 <= plr < 60:
            #         self.plr_distribution["40-60"] += 1
            #     elif 60 <= plr < 80:
            #         self.plr_distribution["60-80"] += 1
            #     elif 80 <= plr <= 100:
            #         self.plr_distribution["80-100"] += 1
            # packetLossRandomNumber = random.randint(0, 100)

            if finalChoiceToOffload:
                chosen_zone_manager, chosen_executor, _, ppo_data = finalChoiceToOffload

                packet_loss_occurred = False
                task_will_be_assigned = False

                packetLossRandomNumber = random.randint(0, 100)

                if packetLossRandomNumber < plr:
                    # print(green_bg("test"))
                    # Packet loss occurred
                    packet_loss_occurred = True
                    task_will_be_assigned = False
                    self.metrics.inc_packet_loss()
                    # print(blue_bg("----------------------------------------------------------------------------"))

                else:
                    # Successful transmission
                    packet_loss_occurred = False
                    task_will_be_assigned = True

                if task_will_be_assigned:
                    # Store zone manager reference
                    self.task_zone_managers[task.id] = chosen_zone_manager

                    if isinstance(chosen_zone_manager, DeepRLZoneManagerPPO):
                        # --- PPO Specific Logic ---
                        state, action, log_prob, value = ppo_data

                        if not chosen_executor.can_offload_task(task):
                            # Immediate failure logic: Node is full or cannot accept
                            reward = Config.NEGATIVE_REWARD
                            self.metrics.add_reward(reward)

                            done = self.clock.get_current_time() >= Config.SimulatorConfig.SIMULATION_DURATION

                            # Store experience immediately since task failed to assign
                            chosen_zone_manager.agent.store_experience(state, action, reward, done, log_prob, value)

                            self.time_step_counter += 1
                            if self.time_step_counter > self.update_timestep == 0 and self.time_step_counter > 0:
                                print(yellow_bg(f"Updating PPO agent at time {current_time}..."))
                                chosen_zone_manager.agent.update()

                            timeout_time = current_time + 1
                            self.schedule_retransmission(task, timeout_time)
                        else:
                            # Delayed reward logic: Save states inside the task for later calculation
                            task.rl_state = state
                            task.rl_action = action
                            task.rl_log_prob = log_prob
                            task.rl_value = value
                            task.rl_zone_manager = chosen_zone_manager

                            chosen_executor.assign_task(task, current_time, self.fixed_fog_nodes)

                if packet_loss_occurred:
                    if task in chosen_executor.tasks:
                        chosen_executor.tasks.remove(task)

                    timeout_time = current_time + Config.SimulatorConfig.TIMEOUT_TIME
                    self.schedule_retransmission(task, timeout_time)
            else:
                self.metrics.inc_no_device_found_to_run_becauseOf_Noise()
                timeout_time = current_time + 1
                self.schedule_retransmission(task, timeout_time)
        else:
            if task.creator.can_offload_task(task):
                task.creator.assign_task(task, current_time, self.fixed_fog_nodes)
            else:
                self.offload_to_cloud(task, current_time, partitions, self.cloud_node)

    def execute_tasks_for_one_step(self):
        """
        Overrides the base execute_tasks_for_one_step to handle PPO's specific
        Replay Buffer storage needs (log_prob and value) upon task completion.
        """
        executed_tasks: List[Task] = []
        merged_nodes: Dict[str, NodeABC] = {
            **self.mobile_fog_nodes,
            **self.user_nodes,
            **self.fixed_fog_nodes,
            self.cloud_node.id: self.cloud_node,
        }

        for node_id, node in merged_nodes.items():
            tasks = node.execute_tasks(self.clock.get_current_time(), self.fixed_fog_nodes)
            executed_tasks.extend(tasks)

            for task in tasks:
                zone_manager = self.task_zone_managers.get(task.id)
                if zone_manager:
                    all_fog_nodes = {**zone_manager.fixed_fog_nodes, **zone_manager.mobile_fog_nodes}
                    loads = [len(n.tasks) for n in all_fog_nodes.values() if n.can_offload_task(task)]
                    if loads:
                        min_load = min(loads)
                        max_load = max(loads)
                        self.metrics.inc_task_load_diff(task.id, min_load, max_load)

                if task.is_hard:
                    self.metrics.inc_local_hard_execution()
                else:
                    if isinstance(task.executor, (FixedFogNode, MobileFogNode)) and (
                            task.creator.id != task.executor.id or Config.ZoneManagerConfig.DEFAULT_ALGORITHM == Config.ZoneManagerConfig.ALGORITHM_ONLY_FOG):
                        self.metrics.inc_fog_execution()
                    elif task.creator.id == task.executor.id:
                        self.metrics.inc_local_execution()
                    elif isinstance(task.executor, CloudNode):
                        self.metrics.inc_cloud_tasks()

                    if hasattr(task, 'rl_state') and hasattr(task, 'rl_action'):
                        rl_zm = getattr(task, 'rl_zone_manager', None)
                        if rl_zm and isinstance(rl_zm, DeepRLZoneManagerPPO):

                            # 1. Calculate REAL reward using the new compute function
                            all_fogs_list = list(self.fixed_fog_nodes.values()) + list(self.mobile_fog_nodes.values())
                            real_reward = rl_zm.env._compute_reward(task, task.executor, all_fogs_list)
                            self.metrics.add_reward(real_reward)

                            # 2. Store the actual experience including specific PPO outputs
                            done = self.clock.get_current_time() >= Config.SimulatorConfig.SIMULATION_DURATION

                            rl_zm.agent.store_experience(
                                task.rl_state,
                                task.rl_action,
                                real_reward,
                                done,
                                task.rl_log_prob,
                                task.rl_value
                            )

                            # 3. Step the timer and update network if needed
                            self.time_step_counter += 1
                            if self.time_step_counter > self.update_timestep == 0 and self.time_step_counter > 0:
                                print(yellow_bg(f"Updating PPO agent at time {self.clock.get_current_time()}..."))
                                rl_zm.agent.update()
                    # -----------------------------------------------------------------

                if task.is_deadline_missed:
                    missed_info = {
                        'task_id': task.id,
                        'release_time': task.release_time,
                        'deadline': task.deadline,
                        'exec_time': task.exec_time,
                        'finish_time': task.finish_time,
                        'executor_id': task.executor.id,
                        'data_size': task.dataSize,
                        'deadline_diff': task.finish_time - task.deadline
                    }
                    self.missed_deadline_data.append(missed_info)

                    if task.is_hard:
                        self.metrics.inc_hard_deadline_miss()
                    else:
                        self.metrics.inc_deadline_miss()
                else:
                    success_task_info = {
                        'task_id': task.id,
                        'release_time': task.release_time,
                        'deadline': task.deadline,
                        'exec_time': task.exec_time,
                        'finish_time': task.finish_time,
                        'executor_id': task.executor.id,
                        'data_size': task.dataSize,
                        'deadline_diff': task.finish_time - task.deadline
                    }
                    self.success_deadline_data.append(success_task_info)
                    self.metrics.inc_completed_task()