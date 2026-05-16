import random
from config import Config
from controllers.finalChoiceByAttenuationNoise import FinalChoiceByAttenuationNoise
from controllers.zone_managers.SAC.deep_rl_zone_manager_sac import DeepRLZoneManagerSAC
from models.node.cloud import CloudNode
from NoiseConfigs.utilsFunctions import UtilsFunc

# Import Base
from controllers.simulator import Simulator, calcAttenuation


class SimulatorSAC(Simulator):
    def choose_executor_and_assign(self, zone_manager_offload_task, task, partitions, current_time):
        if len(zone_manager_offload_task) != 0:
            attenuationList = []

            for candidate in zone_manager_offload_task:
                zone_manager, candidate_executor = candidate  # Standard unpacking

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
                attenuationList.append((zone_manager, candidate_executor, attenuation))

            finalChoiceToOffload, plr = self.noise_controller.makeFinalChoice(
                attenuationList, task, partitions, Config.NoiseMethod.DEFAULT_METHOD
            )

            if finalChoiceToOffload:
                chosen_zone_manager, chosen_executor, _ = finalChoiceToOffload

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

                    if isinstance(chosen_zone_manager, DeepRLZoneManagerSAC):
                        state = chosen_zone_manager.env._get_state(task)
                        reward, action = chosen_zone_manager.env._compute_reward2(task, chosen_executor)

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
                            chosen_executor.assign_task(task, current_time)
                            task_assigned = True

                        next_state = chosen_zone_manager.env._get_state(task)
                        chosen_zone_manager.agent.store_experience(state, action, reward, next_state, done=False)

                        chosen_zone_manager.agent.train()

                    else:
                        chosen_executor.assign_task(task, current_time)
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
                task.creator.assign_task(task, current_time)
            else:
                self.offload_to_cloud(task, current_time, partitions, self.cloud_node)