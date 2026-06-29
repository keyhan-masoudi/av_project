from typing import Dict, List
import random
from config import Config
from controllers.finalChoiceByAttenuationNoise import FinalChoiceByAttenuationNoise
from controllers.zone_managers.DDPG.deep_rl_zone_manager_ddpg import DeepRLZoneManager_DDPG
from models.node.cloud import CloudNode
from models.node.fog import FixedFogNode, MobileFogNode
from models.node.base import NodeABC
from models.task import Task
from NoiseConfigs.utilsFunctions import UtilsFunc

# Import the Base Class
from controllers.simulator import Simulator, calcAttenuation

class SimulatorDDPG(Simulator):
    """
    Inherits everything from Simulator.
    Overrides only the methods specific to DDPG logic.
    """

    def find_zone_manager_offload_task(self, zone_managers, task, current_time):
        """
        Overridden because DDPG returns a 'continuous_action' tuple logic.
        """
        # print("-------------------------------------------------------------------------------------")
        zone_manager_offload_task = []
        for zone_manager in zone_managers:
            if zone_manager.can_offload_task(task):
                if hasattr(zone_manager, "propose_candidate"):
                    proposed_zone_manager, proposed_executor, continuous_action = zone_manager.propose_candidate(task, current_time)
                else:
                    proposed_zone_manager = zone_manager
                    proposed_executor = zone_manager.offload_task(task, current_time)
                    continuous_action = None

                candidate = (proposed_zone_manager, proposed_executor, continuous_action)
                if proposed_executor and candidate not in zone_manager_offload_task:
                    zone_manager_offload_task.append(candidate)
        return zone_manager_offload_task

    def choose_executor_and_assign(self, zone_manager_offload_task, task, partitions, current_time):
        """
        Overridden to handle DDPG experience replay storage and specific unpacking.
        """
        if len(zone_manager_offload_task) != 0:
            # print(green_bg("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
            attenuationList = []

            for candidate in zone_manager_offload_task:
                zone_manager, candidate_executor, continuous_action = candidate

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

                attenuationList.append((zone_manager, candidate_executor, attenuation, continuous_action))

            finalChoiceToOffload, plr = self.noise_controller.makeFinalChoice(attenuationList, task, partitions, Config.NoiseMethod.DEFAULT_METHOD)


            if finalChoiceToOffload:
                chosen_zone_manager, chosen_executor, _, chosen_continuous_action = finalChoiceToOffload

                packet_loss_occurred = False
                task_will_be_assigned = False


                # 0 < PLR < 100: Use bandit and simulate packet loss
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

                    if isinstance(chosen_zone_manager, DeepRLZoneManager_DDPG):
                        state = chosen_zone_manager.env._get_state(task, current_time)

                        # Immediate negative reward if the chosen executor cannot accept the task
                        if not chosen_executor.can_offload_task(task):
                            reward = Config.NEGATIVE_REWARD
                            next_state = chosen_zone_manager.env._get_state(task=None, current_time=current_time)
                            chosen_zone_manager.agent.store_experience(
                                state, chosen_continuous_action, reward, next_state, done=False
                            )
                            chosen_zone_manager.agent.train()

                            timeout_time = current_time + 1
                            self.schedule_retransmission(task, timeout_time)
                        else:
                            # Delay reward calculation
                            task.rl_state = state
                            task.rl_continuous_action = chosen_continuous_action
                            task.rl_action = chosen_zone_manager.env.get_action_from_executor(task, chosen_executor)
                            task.rl_zone_manager = chosen_zone_manager
                            chosen_executor.assign_task(task, current_time, self.fixed_fog_nodes)

                    else:
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
        executed_tasks = []
        merged_nodes = {
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

                    # =================================================================
                    # DDPG DELAYED REWARD LOGIC: Task is fully executed.
                    # =================================================================
                    if hasattr(task, 'rl_state') and hasattr(task, 'rl_continuous_action'):
                        rl_zm = getattr(task, 'rl_zone_manager', None)
                        if rl_zm and isinstance(rl_zm, DeepRLZoneManager_DDPG):
                            all_fogs_for_reward = {**self.fixed_fog_nodes, **self.mobile_fog_nodes}

                            real_reward = rl_zm.env._compute_reward(task, task.executor, all_fogs_for_reward)
                            self.metrics.add_reward(real_reward)

                            current_time = self.clock.get_current_time()
                            next_state = rl_zm.env._get_state(task=None, current_time=current_time)

                            # Store the continuous action for Critic training
                            rl_zm.agent.store_experience(
                                task.rl_state,
                                task.rl_continuous_action,
                                real_reward,
                                next_state,
                                done=False
                            )
                            rl_zm.agent.train()
                    # =================================================================

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