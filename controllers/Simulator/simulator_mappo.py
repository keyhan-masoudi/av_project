import numpy as np
import random
from config import Config
from controllers.simulator import Simulator, calcAttenuation, red_bg, green_bg
from controllers.zone_managers.MAPPO.mappo_controller import MAPPOController
from controllers.zone_managers.MAPPO.deep_rl_zone_manager_mappo import DeepRLZoneManagerMAPPO
from controllers.maddpg_utils import get_agent_state, get_action_mask, compute_agent_reward
from models.node.cloud import CloudNode
from NoiseConfigs.utilsFunctions import UtilsFunc
from utils.enums import Layer

class SimulatorMAPPO(Simulator):
    def __init__(self, loader, clock, cloud):
        super().__init__(loader, clock, cloud)
        self.mappo_controller: MAPPOController = None
        self.agents = []

    def init_simulation(self):
        super().init_simulation()
        self.agents = list(self.zone_managers.values())
        num_agents = len(self.agents)

        # Initialize the centralized MAPPO controller
        self.mappo_controller = MAPPOController(num_agents=num_agents, local_state_dim=28, action_dim=5)

        for i, agent in enumerate(self.agents):
            agent.agent_id = i
            agent.controller = self.mappo_controller

    def get_k_nearest_fogs(self, task_creator, k=3):
        all_fogs = list(self.fixed_fog_nodes.values()) + list(self.mobile_fog_nodes.values())
        fogs_with_dist = [
            (fog, ((fog.x - task_creator.x) ** 2 + (fog.y - task_creator.y) ** 2) ** 0.5)
            for fog in all_fogs
        ]
        fogs_with_dist.sort(key=lambda x: x[1])
        return [f[0] for f in fogs_with_dist[:k]]

    def start_simulation(self):
        self.init_simulation()
        partitions = UtilsFunc.load_partitions("generated_hex_partitions")
        self.load_cached_traffic()
        self.load_cached_vehicle_traffic()
        self.load_cached_weather()
        self.load_cached_spatial_grid()
        self.load_cached_future_predictions()

        while (current_time := self.clock.get_current_time()) < Config.SimulatorConfig.SIMULATION_DURATION:
            print(red_bg(f"current_time:{current_time}"))
            time_int = int(current_time)

            # Update environmental conditions
            cached_data_str_keys = self.traffic_cache.get(time_int, {})
            traffic_data = {p: cached_data_str_keys[p.__class__.__name__] for p in partitions if p.__class__.__name__ in cached_data_str_keys}
            self.update_weather_from_cache(current_time)
            for partition in partitions:
                partition.update_traffic_status(traffic_data)

            soft_tasks = self.load_soft_tasks(current_time)
            self.load_hard_tasks(current_time)

            user_possible_zones = self.assign_mobile_nodes_to_zones(self.user_nodes, layer=Layer.USER)
            mobile_possible_zones = self.assign_mobile_nodes_to_zones(self.mobile_fog_nodes, layer=Layer.FOG)
            merged_possible_zones = {**user_possible_zones, **mobile_possible_zones}

            self.handle_retransmissions(merged_possible_zones, current_time, partitions)

            # Main MAPPO Logic
            for creator_id, tasks in soft_tasks.items():
                if not tasks: continue
                participating_managers = merged_possible_zones.get(creator_id, [])

                if not participating_managers:
                    for task in tasks:
                        self.handle_no_zone_manager(task, current_time, partitions)
                    continue

                for task in tasks:
                    self.metrics.inc_total_tasks()

                    # 1. Get States and Masks to form the Global State
                    current_states = {zm.agent_id: get_agent_state(task, self) for zm in participating_managers}
                    ordered_local_states = [current_states.get(i, np.zeros(self.mappo_controller.local_state_dim)) for i in range(self.mappo_controller.num_agents)]

                    current_masks = {zm.agent_id: get_action_mask(task, self) for zm in participating_managers}
                    ordered_masks = [current_masks.get(i, np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)) for i in range(self.mappo_controller.num_agents)]

                    global_state = np.concatenate(ordered_local_states)

                    # 2. Get actions from the MAPPO network
                    actions, log_probs, value = self.mappo_controller.select_actions(ordered_local_states, global_state, ordered_masks)

                    proposals = {}
                    for zm in participating_managers:
                        action = actions[zm.agent_id]
                        if action == 0: executor = task.creator
                        elif action == 1: executor = self.cloud_node
                        elif action in [2, 3, 4]:
                            fog_idx = action - 2
                            nearest_fogs = self.get_k_nearest_fogs(task.creator, k=3)
                            executor = nearest_fogs[fog_idx] if fog_idx < len(nearest_fogs) else task.creator
                        else: executor = task.creator

                        proposals[zm.agent_id] = executor

                    # 3. Evaluate environmental noise and make the final decision
                    final_executor, chosen_agent_id, status = self.choose_executor_with_noise(proposals, task, partitions, current_time)

                    if status == "SUCCESS":
                        if not final_executor.can_offload_task(task):
                            # Immediate penalty for invalid action
                            self.mappo_controller.store_experience(ordered_local_states[chosen_agent_id], global_state, actions[chosen_agent_id], log_probs[chosen_agent_id], value, Config.NEGATIVE_REWARD)
                            self.schedule_retransmission(task, current_time + 1)
                        else:
                            # Temporary storage in the task for delayed reward calculation
                            task.mappo_local_state = ordered_local_states[chosen_agent_id]
                            task.mappo_global_state = global_state
                            task.mappo_action = actions[chosen_agent_id]
                            task.mappo_log_prob = log_probs[chosen_agent_id]
                            task.mappo_value = value
                            task.mappo_chosen_agent_id = chosen_agent_id

                            final_executor.assign_task(task, current_time, self.fixed_fog_nodes)
                            self.task_zone_managers[task.id] = self.agents[chosen_agent_id]

                    elif status in ["PACKET_LOSS", "LQE_REJECTED"]:
                        penalty = -2.0 if status == "PACKET_LOSS" else -0.5
                        self.mappo_controller.store_experience(ordered_local_states[chosen_agent_id], global_state, actions[chosen_agent_id], log_probs[chosen_agent_id], value, penalty)
                        delay_time = Config.SimulatorConfig.TIMEOUT_TIME if status == "PACKET_LOSS" else 1
                        self.schedule_retransmission(task, current_time + delay_time)
                    else:
                        self.schedule_retransmission(task, current_time + 1)

            # Execute tasks and handle completed ones
            self.execute_tasks_for_one_step()
            self.update_graph()
            self.metrics.flush()
            self.metrics.add_data(current_time)

            # Periodic Batch Size update for the network
            self.mappo_controller.train(batch_size=64)

        # End of simulation
        self.drop_not_completed_tasks()

        hardTasks = "withHardTasks" if Config.SimulatorConfig.ENABLE_HARD_TASKS else "withoutHardTasks"
        parallel = "withParallel" if Config.SimulatorConfig.BASELINE_PARALLEL_FREQUENCY else "withoutParallel"

        self.save_missed_deadlines_to_excel(
            f"missed_deadlines_report_{Config.ZoneManagerConfig.DEFAULT_ALGORITHM}_{Config.NoiseMethod.DEFAULT_METHOD}_{Config.NoiseConfig.DEFAULT_THRESHOLD}_{Config.TrafficNoise.DEFAULT_TrafficNoiseLevel}"
            f"_{Config.AttenuationLevel.DEFAULT_AttenuationLevelName}_{Config.City.DEFAULT_CITY}_{hardTasks}_{parallel}.csv")

        self.save_success_deadlines_to_excel(
            f"success_deadlines_report_{Config.ZoneManagerConfig.DEFAULT_ALGORITHM}_{Config.NoiseMethod.DEFAULT_METHOD}_{Config.NoiseConfig.DEFAULT_THRESHOLD}_{Config.TrafficNoise.DEFAULT_TrafficNoiseLevel}"
            f"_{Config.AttenuationLevel.DEFAULT_AttenuationLevelName}_{Config.City.DEFAULT_CITY}_{hardTasks}_{parallel}.csv")

        self.metrics.save_to_excel(
            f"final_metrics_summary_{Config.ZoneManagerConfig.DEFAULT_ALGORITHM}_{Config.NoiseMethod.DEFAULT_METHOD}_{Config.NoiseConfig.DEFAULT_THRESHOLD}_{Config.TrafficNoise.DEFAULT_TrafficNoiseLevel}"
            f"_{Config.AttenuationLevel.DEFAULT_AttenuationLevelName}_{Config.City.DEFAULT_CITY}_{hardTasks}_{parallel}.xlsx")

        self.metrics.save_convergence_to_csv(
            f"convergence_{Config.ZoneManagerConfig.DEFAULT_ALGORITHM}_{Config.NoiseMethod.DEFAULT_METHOD}_{Config.NoiseConfig.DEFAULT_THRESHOLD}_{Config.TrafficNoise.DEFAULT_TrafficNoiseLevel}"
            f"_{Config.AttenuationLevel.DEFAULT_AttenuationLevelName}_{Config.City.DEFAULT_CITY}_{hardTasks}_{parallel}.csv")


    def choose_executor_with_noise(self, proposals, task, partitions, current_time):
        if not proposals:
            return None, -1, "NO_PROPOSALS"

        attenuation_list = []
        for agent_id, executor in proposals.items():
            if not executor: continue
            intersecting_partitions = UtilsFunc().find_line_intersections(
                (task.creator.x, task.creator.y), (executor.x, executor.y), partitions
            )

            if isinstance(executor, CloudNode):
                attenuation = self.calcAttForCloud(task, intersecting_partitions)
            elif len(intersecting_partitions) > 0:
                attenuation = calcAttenuation(task, executor, intersecting_partitions)
            else:
                attenuation = 0

            zone_manager = self.agents[agent_id]
            attenuation_list.append((zone_manager, executor, attenuation))

        if not attenuation_list:
            return None, -1, "NO_PROPOSALS"

        final_choice, plr = self.noise_controller.makeFinalChoice(attenuation_list, task, partitions, Config.NoiseMethod.DEFAULT_METHOD)

        if final_choice:
            chosen_zone_manager, chosen_executor, _ = final_choice
            packetLossRandomNumber = random.randint(0, 100)

            if packetLossRandomNumber < plr:
                self.metrics.inc_packet_loss()
                if task in chosen_executor.tasks:
                    chosen_executor.tasks.remove(task)
                return None, chosen_zone_manager.agent_id, "PACKET_LOSS"
            else:
                return chosen_executor, chosen_zone_manager.agent_id, "SUCCESS"
        else:
            self.metrics.inc_no_device_found_to_run_becauseOf_Noise()
            return None, -1, "LQE_REJECTED"


    def handle_retransmissions(self, merged_zones, current_time, partitions=None):
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
            self.offload_to_cloud(task, current_time, partitions, self.cloud_node)

    def offload_to_cloud(self, task, current_time, partitions=None, cloud_node=None):
        if partitions is None: partitions = []
        if cloud_node is None: cloud_node = self.cloud_node

        if self.cloud_node.can_offload_task(task):
            attenuationList = []
            intersecting_partitions = UtilsFunc().find_line_intersections(
                (task.creator.x, task.creator.y),
                (self.cloud_node.x, self.cloud_node.y),
                partitions
            )

            attenuation = self.calcAttForCloud(task, intersecting_partitions)
            attenuationList.append((None, cloud_node, attenuation, None))

            finalChoiceToOffload, plr = self.noise_controller.makeFinalChoice(
                attenuationList, task, partitions, Config.NoiseMethod.DEFAULT_METHOD
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

    def execute_tasks_for_one_step(self):
        merged_nodes = {**self.mobile_fog_nodes, **self.user_nodes, **self.fixed_fog_nodes, self.cloud_node.id: self.cloud_node}

        for node_id, node in merged_nodes.items():
            tasks = node.execute_tasks(self.clock.get_current_time(), self.fixed_fog_nodes)
            for task in tasks:
                
                if task.is_hard:
                    self.metrics.inc_local_hard_execution()
                else:
                    from models.node.fog import FixedFogNode, MobileFogNode
                    if isinstance(task.executor, (FixedFogNode, MobileFogNode)) and (task.creator.id != task.executor.id or Config.ZoneManagerConfig.DEFAULT_ALGORITHM == Config.ZoneManagerConfig.ALGORITHM_ONLY_FOG):
                        self.metrics.inc_fog_execution()
                    elif task.creator.id == task.executor.id:
                        self.metrics.inc_local_execution()
                    elif isinstance(task.executor, CloudNode):
                        self.metrics.inc_cloud_tasks()

                    if hasattr(task, 'mappo_local_state'):
                        # A trick to use the previous compute_agent_reward function
                        task.maddpg_ordered_states = {task.mappo_chosen_agent_id: task.mappo_local_state}
                        task.maddpg_actions = {task.mappo_chosen_agent_id: task.mappo_action}
                        task.maddpg_chosen_agent_id = task.mappo_chosen_agent_id

                        all_fogs_for_reward = {**self.fixed_fog_nodes, **self.mobile_fog_nodes}
                        real_reward = compute_agent_reward(task, task.executor, all_fogs_for_reward)

                        self.metrics.add_reward(real_reward)

                        # Save data in buffer for the next update after execution is completed
                        self.mappo_controller.store_experience(
                            task.mappo_local_state,
                            task.mappo_global_state,
                            task.mappo_action,
                            task.mappo_log_prob,
                            task.mappo_value,
                            real_reward
                        )

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
                        print(red_bg(
                            f" HARD TASK MISS DETECTED: {task.id}. Release: {task.release_time}, Deadline: {task.deadline}, Finish: {task.finish_time}, Executor: {task.executor.id}"))
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