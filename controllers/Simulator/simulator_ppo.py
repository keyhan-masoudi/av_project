import random
from config import Config
from controllers.finalChoiceByAttenuationNoise import FinalChoiceByAttenuationNoise
from controllers.zone_managers.PPO.deep_rl_zone_manager_PPO import DeepRLZoneManagerPPO
from models.node.cloud import CloudNode
from NoiseConfigs.utilsFunctions import UtilsFunc

# Import Base
from controllers.simulator import Simulator, calcAttenuation, yellow_bg

class SimulatorPPO(Simulator):

    def __init__(self, loader, clock, cloud):
        # 1. Call Parent Init
        super().__init__(loader, clock, cloud)

        # 2. Add PPO Specific attributes
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

                        done = self.clock.get_current_time() >= Config.SimulatorConfig.SIMULATION_DURATION

                        # PPO Store Experience
                        chosen_zone_manager.agent.store_experience(state, action, reward, done, log_prob, value)
                        self.time_step_counter += 1

                        if self.time_step_counter % self.update_timestep == 0 and self.time_step_counter > 0:
                            print(yellow_bg(f"Updating PPO agent at time {current_time}..."))
                            chosen_zone_manager.agent.update()

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
