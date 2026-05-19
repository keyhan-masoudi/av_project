from typing import Dict, List
import random
from config import Config
from controllers.finalChoiceByAttenuationNoise import FinalChoiceByAttenuationNoise
from controllers.zone_managers.DDPG.deep_rl_zone_manager_ddpg import DeepRLZoneManager_DDPG
from models.node.cloud import CloudNode
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
                        state = chosen_zone_manager.env._get_state(task)
                        reward, _ = chosen_zone_manager.env._compute_reward2(task, chosen_executor)

                        if not chosen_executor.can_offload_task(task) and (reward > Config.NEGATIVE_REWARD):
                            reward = Config.NEGATIVE_REWARD
                            timeout_time = current_time + 1
                            self.schedule_retransmission(task, timeout_time)
                            task_assigned = False
                        elif reward < Config.NEGATIVE_REWARD:
                            timeout_time = current_time + 1
                            self.schedule_retransmission(task, timeout_time)
                            task_assigned = False
                        else:
                            chosen_executor.assign_task(task, current_time, self.fixed_fog_nodes)
                            task_assigned = True

                        next_state = chosen_zone_manager.env._get_state(task)
                        chosen_zone_manager.agent.store_experience(state, chosen_continuous_action, reward, next_state,
                                                                   done=False)
                        chosen_zone_manager.agent.train()

                    else:
                        chosen_executor.assign_task(task, current_time, self.fixed_fog_nodes)
                        task_assigned = True

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