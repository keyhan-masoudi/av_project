import numpy as np
import random
from collections import defaultdict
from typing import Dict, List

from config import Config
from controllers.finalChoiceByAttenuationNoise import FinalChoiceByAttenuationNoise
from controllers.zone_managers.MADDPG.maddpg_controller import MADDPGController
from controllers.maddpg_utils import get_agent_state, compute_agent_reward, get_action_mask
from controllers.zone_managers.MADDPG.deep_rl_zone_manager_maddpg import DeepRLZoneManagerMADDGP
from models.node.base import NodeABC
from models.node.cloud import CloudNode
from models.task import Task
from NoiseConfigs.utilsFunctions import UtilsFunc
from utils.enums import Layer

# Import Base Class
from controllers.simulator import Simulator, calcAttenuation, red_bg, green_bg


class SimulatorMADDPG(Simulator):
    """
    MADDPG Simulator inheriting from the Base Simulator.
    The main simulation loop is overridden due to the centralized nature of MADDPG action selection.
    """

    def __init__(self, loader, clock, cloud):
        # 1. Call Parent Init
        super().__init__(loader, clock, cloud)

        # 2. MADDPG Specific Attributes
        self.maddpg_controller: MADDPGController = None
        self.agents: List[DeepRLZoneManagerMADDGP] = []
        self.training_step_counter = 0

    def init_simulation(self):
        # 1. Standard initialization (Loads zones, nodes, fixed nodes)
        super().init_simulation()

        # 2. MADDPG Specific Setup
        self.agents = list(self.zone_managers.values())
        num_agents = len(self.agents)
        state_dim = 5
        action_dim = 3

        self.maddpg_controller = MADDPGController(
            num_agents=num_agents,
            state_dims=[state_dim] * num_agents,
            action_dims=[action_dim] * num_agents
        )

        for i, agent in enumerate(self.agents):
            agent.agent_id = i
            agent.controller = self.maddpg_controller

    def start_simulation(self):
        # print("---------------------------------------------------------------------")
        # Initialize
        self.init_simulation()
        partitions = UtilsFunc.load_partitions("generated_hex_partitions")
        neighbors_map = UtilsFunc.find_neighbors(partitions)

        self.load_cached_traffic()

        while (current_time := self.clock.get_current_time()) < Config.SimulatorConfig.SIMULATION_DURATION:
            print(red_bg(f"current_time:{current_time}"))
            self.maddpg_controller.train()

            time_int = int(current_time)

            cached_data_str_keys = self.traffic_cache.get(time_int, {})

            traffic_data = {}
            for p in partitions:
                p_name = p.__class__.__name__
                if p_name in cached_data_str_keys:
                    traffic_data[p] = cached_data_str_keys[p_name]
            # print(f"traffic_data:{traffic_data}")

            # Update Rain (Reusing Parent Logic)
            if int(self.clock.get_current_time()) % 5 == 0:
                self.update_rain_with_constraints(neighbors_map, partitions)

            for partition in partitions:
                partition.update_traffic_status(traffic_data)

            # Load Tasks (Reusing Parent Logic)
            nodes_tasks = self.load_tasks(current_time)

            # Assign Zones (Reusing Parent Logic)
            user_possible_zones = self.assign_mobile_nodes_to_zones(self.user_nodes, layer=Layer.USER)
            mobile_possible_zones = self.assign_mobile_nodes_to_zones(self.mobile_fog_nodes, layer=Layer.FOG)
            merged_possible_zones = {**user_possible_zones, **mobile_possible_zones}

            # MADDPG Specific Retransmission Handling
            self.handle_retransmissions(merged_possible_zones, current_time, partitions)

            # --- MADDPG Main Logic Loop ---
            for creator_id, tasks in nodes_tasks.items():
                if not tasks: continue

                participating_managers = merged_possible_zones.get(creator_id, [])
                if not participating_managers:
                    for task in tasks:
                        self.handle_no_zone_manager(task, current_time, partitions)
                    continue

                for task in tasks:
                    self.metrics.inc_total_tasks()

                    # 1. Get States for ALL agents
                    current_states = {zm.agent_id: get_agent_state(task, self) for zm in participating_managers}
                    ordered_states = [current_states.get(i, np.zeros(self.maddpg_controller.state_dims[i])) for i in
                                      range(self.maddpg_controller.num_agents)]

                    current_masks = {zm.agent_id: get_action_mask(task) for zm in participating_managers}
                    ordered_masks = [current_masks.get(i, np.array([1.0, 1.0, 1.0], dtype=np.float32))
                                     for i in range(self.maddpg_controller.num_agents)]
                    # 2. Select Actions (Centralized)
                    actions = self.maddpg_controller.select_actions(ordered_states, masks=ordered_masks)

                    # 3. Create Proposals based on actions
                    proposals = {zm.agent_id: (zm._get_best_fog_node(task) if actions[zm.agent_id] == 1 else (
                        self.cloud_node if actions[zm.agent_id] == 2 else task.creator)) for zm in
                                 participating_managers}

                    # 4. Choose Executor with Noise
                    final_executor, chosen_agent_id, status = self.choose_executor_with_noise(proposals, task,
                                                                                              partitions, current_time)

                    # 5. Compute Rewards & Assign
                    rewards = np.zeros(self.maddpg_controller.num_agents)

                    if status == "SUCCESS":
                        # اجرای موفقیت آمیز
                        rewards[chosen_agent_id] = compute_agent_reward(task, final_executor, self.fixed_fog_nodes)
                        final_executor.assign_task(task, current_time, self.fixed_fog_nodes)
                        self.task_zone_managers[task.id] = self.agents[chosen_agent_id]

                    elif status == "PACKET_LOSS":
                        # جریمه متوسط برای از دست رفتن پکت (ارسال انجام شده اما ناموفق بوده)
                        rewards[chosen_agent_id] = -2.0
                        for agent_id, executor in proposals.items():
                            if executor and agent_id != chosen_agent_id:
                                rewards[agent_id] = -0.1
                        self.schedule_retransmission(task, current_time + Config.SimulatorConfig.TIMEOUT_TIME)

                    elif status == "LQE_REJECTED":
                        # جریمه نرم (Soft Penalty) برای رد شدن پیشنهاد توسط ارزیاب کیفیت لینک
                        # به جای جریمه -5.0، یک جریمه کوچک -0.5 می‌دهیم تا الگوریتم کلا از Fog فراری نشود
                        for agent_id, executor in proposals.items():
                            if executor:
                                rewards[agent_id] = -0.5
                        self.schedule_retransmission(task, current_time + 1)

                    else:  # status == "NO_PROPOSALS"
                        self.schedule_retransmission(task, current_time + 1)

                    # 6. Store Experience
                    next_states = {zm.agent_id: get_agent_state(task, self) for zm in participating_managers}
                    ordered_next_states = [next_states.get(i, np.zeros(self.maddpg_controller.state_dims[i])) for i in
                                           range(self.maddpg_controller.num_agents)]

                    dones = np.zeros(self.maddpg_controller.num_agents)
                    self.maddpg_controller.store_experience(ordered_states, actions, rewards, ordered_next_states,
                                                            dones)

            # Update Graph (Reusing Parent Logic)
            self.update_graph()

            # End Step (Reusing Parent Logic)
            self.execute_tasks_for_one_step()
            self.metrics.flush()
            self.metrics.add_data(current_time)

        # End Simulation (Reusing Parent Logic)
        self.drop_not_completed_tasks()
        self.save_missed_deadlines_to_excel(
            f"2missed_deadlines_report_{Config.ZoneManagerConfig.DEFAULT_ALGORITHM}_{Config.NoiseMethod.DEFAULT_METHOD}_{Config.NoiseConfig.DEFAULT_THRESHOLD}_{Config.TrafficNoise.DEFAULT_TrafficNoiseLevel}_{Config.AttenuationLevel.DEFAULT_AttenuationLevelName}_{Config.City.DEFAULT_CITY}.csv")
        self.save_success_deadlines_to_excel(
            f"2success_deadlines_report_{Config.ZoneManagerConfig.DEFAULT_ALGORITHM}_{Config.NoiseMethod.DEFAULT_METHOD}_{Config.NoiseConfig.DEFAULT_THRESHOLD}_{Config.TrafficNoise.DEFAULT_TrafficNoiseLevel}_{Config.AttenuationLevel.DEFAULT_AttenuationLevelName}_{Config.City.DEFAULT_CITY}.csv")
        self.metrics.save_to_excel(
            f"2final_metrics_summary_{Config.ZoneManagerConfig.DEFAULT_ALGORITHM}_{Config.NoiseMethod.DEFAULT_METHOD}_{Config.NoiseConfig.DEFAULT_THRESHOLD}_{Config.TrafficNoise.DEFAULT_TrafficNoiseLevel}_{Config.AttenuationLevel.DEFAULT_AttenuationLevelName}_{Config.City.DEFAULT_CITY}.xlsx")

    def choose_executor_with_noise(self, proposals: Dict[int, NodeABC], task: Task, partitions, current_time):
        if not proposals:
            return None, -1, "NO_PROPOSALS"

        attenuation_list = []
        for agent_id, executor in proposals.items():
            if not executor: continue

            intersecting_partitions = UtilsFunc().find_line_intersections(
                (task.creator.x, task.creator.y), (executor.x, executor.y), partitions
            )

            if isinstance(executor, CloudNode):
                # Use Parent's calcAttForCloud
                attenuation = self.calcAttForCloud(task, intersecting_partitions)
            elif len(intersecting_partitions) > 0:
                # Use Global/Parent calcAttenuation
                attenuation = calcAttenuation(task, executor, intersecting_partitions)
            else:
                attenuation = 0

            zone_manager = self.agents[agent_id]
            attenuation_list.append((zone_manager, executor, attenuation))

        if not attenuation_list:
            return None, -1, "NO_PROPOSALS"

        final_choice, plr = self.noise_controller.makeFinalChoice(
            attenuation_list, task, partitions, Config.NoiseMethod.DEFAULT_METHOD, current_time=current_time,
            fixed_fog_nodes=self.fixed_fog_nodes
        )

        if final_choice:
            # Unpacking 3 elements
            chosen_zone_manager, chosen_executor, _ = final_choice

            use_bandit, should_offload = self.noise_controller.adaptive_manager.should_use_bandit(plr)

            packet_loss_occurred = False
            task_will_be_assigned = False

            if not use_bandit and Config.NoiseMethod.DEFAULT_METHOD == Config.NoiseMethod.PROPOSED_METHOD4:
                # PLR is 0 or 100 - direct decision without bandit
                if plr <= 1.0:
                    packet_loss_occurred = False
                    task_will_be_assigned = True
                else:
                    self.metrics.inc_no_device_found_to_run_becauseOf_Noise()
                    return None, -1, "LQE_REJECTED"
            else:
                # 0 < PLR < 100: Use bandit and simulate packet loss
                packetLossRandomNumber = random.randint(0, 100)

                if packetLossRandomNumber < plr:
                    packet_loss_occurred = True
                    task_will_be_assigned = False
                    self.metrics.inc_packet_loss()

                else:
                    packet_loss_occurred = False
                    task_will_be_assigned = True

            if task_will_be_assigned:
                return chosen_executor, chosen_zone_manager.agent_id, "SUCCESS"

            if packet_loss_occurred:
                if task in chosen_executor.tasks:
                    chosen_executor.tasks.remove(task)
                return None, chosen_zone_manager.agent_id, "PACKET_LOSS"

        else:
            self.metrics.inc_no_device_found_to_run_becauseOf_Noise()
            return None, -1, "LQE_REJECTED"

    def handle_retransmissions(self, merged_zones, current_time, partitions=None):
        # Logic specific to MADDPG's handling (as provided in original code)
        tasks_to_retransmit = self.retransmission_tasks.pop(current_time, [])
        for task in tasks_to_retransmit:
            participating_managers = merged_zones.get(task.creator.id, [])
            if not participating_managers:
                self.handle_no_zone_manager(task, current_time, partitions)
                continue

    def handle_no_zone_manager(self, task, current_time, partitions=None):
        if task.creator.can_offload_task(task):
            task.creator.assign_task(task, current_time, self.fixed_fog_nodes)
        else:
            self.offload_to_cloud(task, current_time, partitions,
                                  self.cloud_node)  # Passed empty partitions or handle inside

    def offload_to_cloud(self, task: Task, current_time: float, partitions=None, cloud_node=None):
        # Overridden because MADDPG uses a 4-element tuple in attenuationList (None at end)
        # and parent uses 3-element.
        if partitions is None: partitions = []  # Handle default
        if cloud_node is None: cloud_node = self.cloud_node

        if self.cloud_node.can_offload_task(task):
            attenuationList = []
            intersecting_partitions = UtilsFunc().find_line_intersections(
                (task.creator.x, task.creator.y),
                (self.cloud_node.x, self.cloud_node.y),
                partitions
            )
            # Use Parent's calcAttForCloud
            attenuation = self.calcAttForCloud(task, intersecting_partitions)

            # MADDPG specific tuple (4 elements)
            attenuationList.append((None, cloud_node, attenuation, None))

            finalChoiceToOffload, plr = FinalChoiceByAttenuationNoise().makeFinalChoice(
                attenuationList, task, partitions, Config.NoiseMethod.DEFAULT_METHOD, current_time=current_time,
                fixed_fog_nodes=self.fixed_fog_nodes
            )

            packetLossRandomNumber = random.randint(0, 100)

            if finalChoiceToOffload:
                chosen_zone_manager, chosen_executor, _, _ = finalChoiceToOffload

                if packetLossRandomNumber < plr:
                    self.metrics.inc_packet_loss()
                    if task in chosen_executor.tasks:
                        chosen_executor.tasks.remove(task)
                    timeout_time = current_time + Config.SimulatorConfig.TIMEOUT_TIME
                    self.schedule_retransmission(task, timeout_time)
                else:
                    self.task_zone_managers[task.id] = chosen_zone_manager
                    self.cloud_node.assign_task(task, current_time, self.fixed_fog_nodes)
            else:
                self.metrics.inc_no_device_found_to_run_becauseOf_Noise()
                timeout_time = current_time + 1
                self.schedule_retransmission(task, timeout_time)
        else:
            self.schedule_retransmission(task, 1)
