import os

import numpy as np
import random
from collections import defaultdict
from typing import Dict, List

from NoiseConfigs.noiseConfigGeneralAttribute import NoiseConfigGeneralAttribute
from config import Config
from controllers.loader import Loader
from controllers.metric import MetricsController, green_bg
from models.node.base import MobileNodeABC, NodeABC
from models.node.cloud import CloudNode
from models.node.fog import FixedFogNode, MobileFogNode
from models.node.user import UserNode
from models.task import Task
from utils.clock import Clock
from utils.enums import Layer
import pandas as pd
from NoiseConfigs.noiseConfig import NoiseConfig

from NoiseConfigs.utilsFunctions import UtilsFunc
from controllers.finalChoiceByAttenuationNoise import FinalChoiceByAttenuationNoise

from controllers.zone_managers.MADDPG.maddpg_controller import MADDPGController
from controllers.maddpg_utils import get_agent_state, compute_agent_reward
from controllers.zone_managers.MADDPG.deep_rl_zone_manager_maddpg import DeepRLZoneManagerMADDGP


def blue_bg(text): return f"\033[44m{text}\033[0m"


def red_bg(text): return f"\033[41m{text}\033[0m"


def yellow_bg(text): return f"\033[43m{text}\033[0m"


def calc_attenuation(task, node, intersecting_partitions):
    return UtilsFunc().path_loss_km_ghz(
        d_km=UtilsFunc().distance(task.creator.x, task.creator.y, node.x, node.y) / 1000,
        f_ghz=UtilsFunc().FREQUENCY_GH,
        n=UtilsFunc().get_max_urban_status(intersecting_partitions)
    ) + UtilsFunc().get_max_rain_attenuation(intersecting_partitions)


class SimulatorMADDPG:
    def __init__(self, loader: Loader, clock: Clock, cloud: CloudNode):
        self.metrics: MetricsController = MetricsController()
        self.loader: Loader = loader
        self.cloud_node: CloudNode = cloud
        self.zone_managers: Dict[str, DeepRLZoneManagerMADDGP] = {}
        self.fixed_fog_nodes: Dict[str, FixedFogNode] = {}
        self.mobile_fog_nodes: Dict[str, MobileFogNode] = {}
        self.user_nodes: Dict[str, UserNode] = {}
        self.clock: Clock = clock
        self.task_zone_managers: Dict[str, DeepRLZoneManagerMADDGP] = {}
        self.maddpg_controller: MADDPGController = None
        self.agents: List[DeepRLZoneManagerMADDGP] = []
        self.retransmission_tasks: Dict[float, List[Task]] = defaultdict(list)
        self.training_step_counter = 0
        self.missed_deadline_data: List[Dict] = []
        self.success_deadline_data: List[Dict] = []

    def init_simulation(self):
        self.clock.set_current_time(0)
        self.zone_managers = self.loader.load_zones()
        self.fixed_fog_nodes = self.loader.load_fixed_zones()
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

        self.assign_fixed_nodes()
        self.update_mobile_fog_nodes_coordinate()
        self.update_user_nodes_coordinate()

    def start_simulation(self):
        self.init_simulation()
        partitions = UtilsFunc.load_partitions("generated_hex_partitions")
        neighbors_map = UtilsFunc.find_neighbors(partitions)

        while (current_time := self.clock.get_current_time()) < Config.SimulatorConfig.SIMULATION_DURATION:
            print(red_bg(f"current_time:{current_time}"))
            self.maddpg_controller.train()

            traffic_data = UtilsFunc.recognize_traffic_status(
                f"D:\Abbas\python projects\VANET - Copy\SumoDividedByTime\Outputs2\dataInTime{int(self.clock.get_current_time())}.csv",
                partitions
            )
            if int(self.clock.get_current_time()) % 5 == 0:
                self.update_rain_with_constraints(neighbors_map, partitions)

            for partition in partitions:
                partition.update_traffic_status(traffic_data)

            self.update_graph()

            nodes_tasks = self.load_tasks(current_time)
            user_possible_zones = self.assign_mobile_nodes_to_zones(self.user_nodes, layer=Layer.USER)
            mobile_possible_zones = self.assign_mobile_nodes_to_zones(self.mobile_fog_nodes, layer=Layer.FOG)
            merged_possible_zones: Dict[str, List[DeepRLZoneManagerMADDGP]] = {**user_possible_zones,
                                                                               **mobile_possible_zones}

            self.handle_retransmissions(merged_possible_zones, current_time)

            for creator_id, tasks in nodes_tasks.items():
                if not tasks: continue

                participating_managers = merged_possible_zones.get(creator_id, [])
                if not participating_managers:
                    for task in tasks:
                        self.handle_no_zone_manager(task, current_time)
                    continue

                for task in tasks:
                    self.metrics.inc_total_tasks()

                    current_states = {zm.agent_id: get_agent_state(task, self) for zm in participating_managers}
                    ordered_states = [current_states.get(i, np.zeros(self.maddpg_controller.state_dims[i])) for i in
                                      range(self.maddpg_controller.num_agents)]

                    actions = self.maddpg_controller.select_actions(ordered_states)

                    proposals = {zm.agent_id: (zm._get_best_fog_node(task) if actions[zm.agent_id] == 1 else (
                        self.cloud_node if actions[zm.agent_id] == 2 else task.creator)) for zm in
                                 participating_managers}

                    final_executor, chosen_agent_id = self.choose_executor_with_noise(proposals, task, partitions)

                    rewards = np.zeros(self.maddpg_controller.num_agents)
                    if final_executor:
                        rewards[chosen_agent_id] = compute_agent_reward(task, final_executor, self.fixed_fog_nodes)
                        final_executor.assign_task(task, current_time)
                        # self.metrics.inc_node_tasks(final_executor.id)
                        self.task_zone_managers[task.id] = self.agents[chosen_agent_id]
                    else:
                        for agent_id, executor in proposals.items():
                            if executor: rewards[agent_id] = -5.0
                        self.schedule_retransmission(task, current_time + 1)

                    next_states = {zm.agent_id: get_agent_state(task, self) for zm in participating_managers}
                    ordered_next_states = [next_states.get(i, np.zeros(self.maddpg_controller.state_dims[i])) for i in
                                           range(self.maddpg_controller.num_agents)]

                    dones = np.zeros(self.maddpg_controller.num_agents)
                    self.maddpg_controller.store_experience(ordered_states, actions, rewards, ordered_next_states,
                                                            dones)

                    # self.training_step_counter += 1
                    # if self.training_step_counter % 10 == 0:
                    #     self.maddpg_controller.train()

            self.execute_tasks_for_one_step()
            self.metrics.flush()
            # self.metrics.log_metrics()
            self.metrics.add_data(current_time)

        self.drop_not_completed_tasks()
        self.save_missed_deadlines_to_excel(f"missed_deadlines_report_{Config.ZoneManagerConfig.DEFAULT_ALGORITHM}_{Config.NoiseMethod.DEFAULT_METHOD}_{Config.NoiseConfig.DEFAULT_THRESHOLD}_{Config.TrafficNoise.DEFAULT_TrafficNoiseLevel}_{Config.AttenuationLevel.DEFAULT_AttenuationLevelName}.xlsx")
        self.save_success_deadlines_to_excel(f"success_deadlines_report_{Config.ZoneManagerConfig.DEFAULT_ALGORITHM}_{Config.NoiseMethod.DEFAULT_METHOD}_{Config.NoiseConfig.DEFAULT_THRESHOLD}_{Config.TrafficNoise.DEFAULT_TrafficNoiseLevel}_{Config.AttenuationLevel.DEFAULT_AttenuationLevelName}.xlsx")
        self.metrics.save_to_excel(f"final_metrics_summary_{Config.ZoneManagerConfig.DEFAULT_ALGORITHM}_{Config.NoiseMethod.DEFAULT_METHOD}_{Config.NoiseConfig.DEFAULT_THRESHOLD}_{Config.TrafficNoise.DEFAULT_TrafficNoiseLevel}_{Config.AttenuationLevel.DEFAULT_AttenuationLevelName}.xlsx")

    def choose_executor_with_noise(self, proposals: Dict[int, NodeABC], task: Task, partitions):
        if not proposals:
            return None, -1

        attenuation_list = []
        for agent_id, executor in proposals.items():
            if not executor: continue

            intersecting_partitions = UtilsFunc().find_line_intersections(
                (task.creator.x, task.creator.y), (executor.x, executor.y), partitions
            )

            if isinstance(executor, CloudNode):
                attenuation = self.calc_att_for_cloud(task, intersecting_partitions)
            elif len(intersecting_partitions) > 0:
                attenuation = calc_attenuation(task, executor, intersecting_partitions)
            else:
                attenuation = 0  # Local execution

            zone_manager = self.agents[agent_id]
            attenuation_list.append((zone_manager, executor, attenuation))

        if not attenuation_list:
            return None, -1

        final_choice, plr = FinalChoiceByAttenuationNoise().makeFinalChoice(
            attenuation_list, task, partitions, Config.NoiseMethod.DEFAULT_METHOD
        )

        if final_choice and random.randint(0, 100) >= plr:
            chosen_zone_manager, chosen_executor, _ = final_choice
            return chosen_executor, chosen_zone_manager.agent_id
        else:
            if final_choice:
                self.metrics.inc_packet_loss()
            else:
                self.metrics.inc_no_device_found_to_run_becauseOf_Noise()
            return None, -1

    def update_rain_with_constraints(self, neighbors_map, partitions):
        """
        Updates the rain status of each partition based on its neighbors' status
        to ensure the difference is not more than 2 units.
        """
        for p in partitions:
            neighbors = neighbors_map.get(p, [])
            if not neighbors:
                # If a partition has no neighbors, it can change freely
                p.change_rainStatus()
                continue

            # Find the min and max rain unit among neighbors
            neighbor_units = [
                NoiseConfigGeneralAttribute.Rain_class_to_unit[n.rainStatus.__class__.__name__]
                for n in neighbors
            ]
            min_neighbor_unit = min(neighbor_units)
            max_neighbor_unit = max(neighbor_units)

            # Determine the allowed range for the new unit of the current partition 'p'
            # The new unit must be at most 2 units away from the furthest neighbor.
            min_allowed_unit = max(0, max_neighbor_unit - 2)
            max_allowed_unit = min(len(NoiseConfigGeneralAttribute.Rain_options) - 1, min_neighbor_unit + 2)

            # Create a list of valid rain options
            allowed_options = []
            if min_allowed_unit <= max_allowed_unit:
                for unit in range(min_allowed_unit, max_allowed_unit + 1):
                    allowed_options.append(NoiseConfigGeneralAttribute.Rain_options[unit])

            # If there are valid options, choose one randomly and update
            if allowed_options:
                new_rain_status_str = random.choice(allowed_options)
                p.rainStatus = eval(new_rain_status_str)

    def handle_retransmissions(self, merged_zones, current_time):
        tasks_to_retransmit = self.retransmission_tasks.pop(current_time, [])
        for task in tasks_to_retransmit:
            participating_managers = merged_zones.get(task.creator.id, [])
            if not participating_managers:
                self.handle_no_zone_manager(task, current_time)
                continue

    def handle_no_zone_manager(self, task, current_time):
        if task.creator.can_offload_task(task):
            task.creator.assign_task(task, current_time)
            # self.metrics.inc_local_execution()
        else:
            self.offload_to_cloud(task, current_time)

    def schedule_retransmission(self, task: Task, scheduled_time: float):
        self.retransmission_tasks[scheduled_time].append(task)
        if task.executor and task in task.executor.tasks:
            task.executor.tasks.remove(task)

    def calc_att_for_cloud(self, task, intersecting_partitions):
        nearest_node_id = min(self.fixed_fog_nodes, key=lambda nid: UtilsFunc.distance(
            self.fixed_fog_nodes[nid].x, self.fixed_fog_nodes[nid].y, task.creator.x, task.creator.y
        ))
        nearest_node = self.fixed_fog_nodes[nearest_node_id]
        return calc_attenuation(task, nearest_node, intersecting_partitions)

    def offload_to_cloud(self, task: Task, current_time: float, partitions, cloud_node):
        if self.cloud_node.can_offload_task(task):
            attenuationList = []
            intersecting_partitions = UtilsFunc().find_line_intersections(
                (task.creator.x, task.creator.y),
                (self.cloud_node.x, self.cloud_node.y),
                partitions
            )
            attenuation = self.calc_att_for_cloud(task, intersecting_partitions)
            attenuationList.append((None, cloud_node, attenuation, None))

            finalChoiceToOffload, plr = FinalChoiceByAttenuationNoise().makeFinalChoice(attenuationList,
                                                                                        task,
                                                                                        partitions,
                                                                                        Config.NoiseMethod.DEFAULT_METHOD)

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
                    # self.metrics.inc_node_tasks(chosen_executor.id)
                    self.cloud_node.assign_task(task, current_time)
            else:
                self.metrics.inc_no_device_found_to_run_becauseOf_Noise()
                timeout_time = current_time + 1
                self.schedule_retransmission(task, timeout_time)
        else:
            self.schedule_retransmission(task, 1)

    def load_tasks(self, current_time: float) -> Dict[str, List[Task]]:
        tasks: Dict[str, List[Task]] = defaultdict(list)
        for creator_id, creator_tasks in self.loader.load_nodes_tasks(current_time).items():
            creator = self.user_nodes.get(creator_id) or self.mobile_fog_nodes.get(creator_id)
            if creator:
                for task in creator_tasks:
                    task.creator = creator
                    tasks[creator_id].append(task)
        return tasks

    def execute_tasks_for_one_step(self):
        executed_tasks: List[Task] = []
        merged_nodes: Dict[str, NodeABC] = {**self.mobile_fog_nodes, **self.user_nodes, **self.fixed_fog_nodes,
                                            self.cloud_node.id: self.cloud_node}
        for node_id, node in merged_nodes.items():
            tasks = node.execute_tasks(self.clock.get_current_time(), self.fixed_fog_nodes)
            executed_tasks.extend(tasks)
            for task in tasks:
                if isinstance(task.executor, (FixedFogNode, MobileFogNode)):
                    self.metrics.inc_fog_execution()
                elif task.creator.id == task.executor.id:
                    self.metrics.inc_local_execution()
                elif isinstance(task.executor, CloudNode):
                    self.metrics.inc_cloud_tasks()

                if task.is_deadline_missed:
                    # print(blue_bg(
                    #     f"DEADLINE MISS: {task.id}, Executor: {task.executor.id}, Diff: {task.finish_time - task.deadline}"))
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

    def update_graph(self):
        self.clock.tick()
        self.update_user_nodes_coordinate()
        self.update_mobile_fog_nodes_coordinate()

    def assign_mobile_nodes_to_zones(self, mobile_nodes: dict, layer: Layer) -> Dict[
        str, List[DeepRLZoneManagerMADDGP]]:
        nodes_possible_zones: Dict[str, List[DeepRLZoneManagerMADDGP]] = defaultdict(list)
        for z_id, zone_manager in self.zone_managers.items():
            nodes: List[MobileNodeABC] = []
            for n_id, mobile_node in mobile_nodes.items():
                if zone_manager.zone.is_in_coverage(mobile_node.x, mobile_node.y):
                    nodes.append(mobile_node)
                    nodes_possible_zones[n_id].append(zone_manager)
            if layer == Layer.FOG:
                zone_manager.set_mobile_fog_nodes(nodes)
        return nodes_possible_zones

    def assign_fixed_nodes(self):
        for z_id, zone_manager in self.zone_managers.items():
            fixed_nodes: List[FixedFogNode] = []
            for n_id, fixed_node in self.fixed_fog_nodes.items():
                if zone_manager.zone.is_in_coverage(fixed_node.x, fixed_node.y):
                    fixed_nodes.append(fixed_node)
            zone_manager.add_fixed_fog_nodes(fixed_nodes)

    def update_mobile_fog_nodes_coordinate(self) -> None:
        new_nodes_data = self.loader.load_mobile_fog_nodes(self.clock.get_current_time())
        self.mobile_fog_nodes = self._update_nodes_coordinate(self.mobile_fog_nodes, new_nodes_data)

    def update_user_nodes_coordinate(self) -> None:
        new_nodes_data = self.loader.load_user_nodes(self.clock.get_current_time())
        self.user_nodes = self._update_nodes_coordinate(self.user_nodes, new_nodes_data)

    @staticmethod
    def _update_nodes_coordinate(old_nodes: dict, new_nodes: dict) -> dict:
        data: Dict = {}
        for n_id, new_node in new_nodes.items():
            if n_id not in old_nodes:
                node = new_node
            else:
                node = old_nodes[n_id]
                node.x, node.y, node.angle, node.speed = new_node.x, new_node.y, new_node.angle, new_node.speed
            data[n_id] = node
        return data

    def drop_not_completed_tasks(self) -> List[Task]:
        left_tasks: list[Task] = []
        merged_nodes: Dict[str, NodeABC] = {**self.mobile_fog_nodes, **self.user_nodes, **self.fixed_fog_nodes,
                                            self.cloud_node.id: self.cloud_node}
        for node_id, node in merged_nodes.items():
            left_tasks.extend(node.tasks)
            for _ in node.tasks:
                self.metrics.inc_deadline_miss()
        return left_tasks

    def save_missed_deadlines_to_excel(self, filename: str = "missed_deadlines.xlsx"):
        output_dir = "Results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directory '{output_dir}' created.")
        full_path = os.path.join(output_dir, filename)

        df = pd.DataFrame(self.missed_deadline_data)
        try:
            df.to_excel(full_path, index=False)
            print(green_bg(f"Successfully saved missed deadline data to {filename}"))
        except Exception as e:
            print(red_bg(f"Error saving to Excel file: {e}"))

    def save_success_deadlines_to_excel(self, filename: str = "success_deadlines.xlsx"):
        output_dir = "Results_Success"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directory '{output_dir}' created.")
        full_path = os.path.join(output_dir, filename)

        df = pd.DataFrame(self.success_deadline_data)

        try:
            df.to_excel(full_path, index=False)
            print(green_bg(f"Successfully saved success deadline data to {filename}"))
        except Exception as e:
            print(red_bg(f"Error saving to Excel file: {e}"))
