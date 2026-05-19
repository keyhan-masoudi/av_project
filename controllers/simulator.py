import glob
import random
from collections import defaultdict
from typing import Dict, List, Optional

from NoiseConfigs.noiseConfigGeneralAttribute import NoiseConfigGeneralAttribute
from NoiseConfigs.utilsFunctions import UtilsFunc
from NoiseConfigs.noiseConfig import NoiseConfig
from config import Config
from controllers.finalChoiceByAttenuationNoise import FinalChoiceByAttenuationNoise
from controllers.loader import Loader
from controllers.metric import MetricsController
from controllers.zone_managers.base import ZoneManagerABC
from controllers.zone_managers.deepRL.deep_rl_zone_manager import DeepRLZoneManager
from models.node.base import MobileNodeABC, NodeABC
from models.node.cloud import CloudNode
from models.node.fog import FixedFogNode
from models.node.fog import MobileFogNode
from models.node.user import UserNode
from models.task import Task
from utils.clock import Clock
from utils.enums import Layer
import sys
import os
import pickle
import pandas as pd

sys.path.append(os.path.abspath(Config.Paths.NoiseConfigsPath))


def yellow_bg(text):
    return f"\033[43m{text}\033[0m"


def red_bg(text):
    return f"\033[41m{text}\033[0m"


def blue_bg(text):
    return f"\033[44m{text}\033[0m"


def green_bg(text):
    return f"\033[42m{text}\033[0m"


# note: check again
def logAttenuation(attenuationList):
    sampleList = []
    for i in range(0, len(attenuationList)):
        sampleList.append(attenuationList[i])
    print(yellow_bg("attenuationList") + f":{len(sampleList)}")


def calcAttenuation(task, node, intersecting_partitions):
    """
    node : nearest_node or executor node
    """
    return UtilsFunc().path_loss_km_ghz(
        d_km=UtilsFunc().distance(task.creator.x, task.creator.y,
                                  node.x, node.y) / 1000,
        f_ghz=UtilsFunc().FREQUENCY_GH,
        n=UtilsFunc().get_max_urban_status(intersecting_partitions)
    ) + UtilsFunc().get_max_rain_attenuation(intersecting_partitions)


class Simulator:
    def __init__(self, loader: Loader, clock: Clock, cloud: CloudNode):
        self.metrics: MetricsController = MetricsController()
        self.loader: Loader = loader
        self.cloud_node: CloudNode = cloud
        self.zone_managers: Dict[str, ZoneManagerABC] = {}
        self.fixed_fog_nodes: Dict[str, FixedFogNode] = {}
        self.mobile_fog_nodes: Dict[str, MobileFogNode] = {}
        self.user_nodes: Dict[str, UserNode] = {}
        self.clock: Clock = clock
        self.task_zone_managers: Dict[str, ZoneManagerABC] = {}
        self.retransmission_tasks: Dict[float, List[Task]] = {}
        self.missed_deadline_data: List[Dict] = []
        self.success_deadline_data: List[Dict] = []
        self.traffic_cache: Dict[int, any] = {}
        self.traffic_predictions = defaultdict(dict)
        self.noise_controller = FinalChoiceByAttenuationNoise()

    def init_simulation(self):
        self.clock.set_current_time(Config.SimulatorConfig.SIMULATION_START_TIME)
        self.zone_managers = self.loader.load_zones()
        self.fixed_fog_nodes = self.loader.load_fixed_zones()
        self.assign_fixed_nodes()
        self.update_mobile_fog_nodes_coordinate()
        self.update_user_nodes_coordinate()
        # check this line, i think there is no need for this attribute
        # self.historical_traffic_features.clear()
        self.traffic_predictions.clear()
        self.load_all_predictions_from_csv("data/prediction_data")
        # For zone managers that use deep RL, the simulator reference is set.
        for zm in self.zone_managers.values():
            if hasattr(zm, "set_simulator"):
                zm.set_simulator(self)

    def schedule_retransmission(self, task: Task, scheduled_time: float):
        if scheduled_time not in self.retransmission_tasks:
            self.retransmission_tasks[scheduled_time] = []
        self.retransmission_tasks[scheduled_time].append(task)

    def assign_fixed_nodes(self):
        for z_id, zone_manager in self.zone_managers.items():
            fixed_nodes: List[FixedFogNode] = []
            for n_id, fixed_node in self.fixed_fog_nodes.items():
                if zone_manager.zone.is_in_coverage(fixed_node.x, fixed_node.y):
                    fixed_nodes.append(fixed_node)
            zone_manager.add_fixed_fog_nodes(fixed_nodes)

    def logTrafficStatus(self, partitions):
        for partition in partitions:
            print(
                f"Partition at ({partition.centerX}, {partition.centerY}): Traffic Status = {partition.trafficStatus.__class__.__name__} rain Status = {partition.rainStatus.__class__.__name__}")

    def retransmission(self, zone_managers, current_time, partitions):
        tasks_to_retransmit = []
        for scheduled_time in list(self.retransmission_tasks.keys()):
            if scheduled_time <= current_time:
                tasks_to_retransmit.extend(self.retransmission_tasks.pop(scheduled_time))

        if tasks_to_retransmit:
            for task in tasks_to_retransmit:
                possible_zone_managers = self.find_zone_manager_offload_task(zone_managers, task, current_time)
                if self.choose_executor_and_assign(possible_zone_managers, task, partitions, current_time):
                    continue

    def find_zone_manager_offload_task(self, zone_managers, task, current_time):
        zone_manager_offload_task = []
        for zone_manager in zone_managers:
            # print(f"zone_manager:{zone_manager.zone}")
            if zone_manager.can_offload_task(task):
                # has_offloaded = True

                # assign task
                if hasattr(zone_manager, "propose_candidate"):
                    proposed_zone_manager, proposed_executor = zone_manager.propose_candidate(task,
                                                                                              current_time)
                    # print(f"proposed_executor:{proposed_executor}")
                else:
                    proposed_zone_manager = zone_manager
                    proposed_executor = zone_manager.offload_task(task, current_time)

                if proposed_executor not in zone_manager_offload_task:
                    if proposed_executor:
                        zone_manager_offload_task.append((proposed_zone_manager, proposed_executor))
        return zone_manager_offload_task

    def calcAttForCloud(self, task, intersecting_partitions):
        nearest_node_id = min(self.fixed_fog_nodes.keys(),
                              key=lambda node_id: ((self.fixed_fog_nodes[node_id].x - task.creator.x) ** 2 +
                                                   (self.fixed_fog_nodes[
                                                        node_id].y - task.creator.y) ** 2) ** 0.5)
        nearest_node = self.fixed_fog_nodes[nearest_node_id]
        return calcAttenuation(task, nearest_node, intersecting_partitions)

    def choose_executor_and_assign(self, zone_manager_offload_task, task, partitions, current_time):
        # if any ZM suggest any device to offload
        if len(zone_manager_offload_task) != 0:
            attenuationList = []

            for candidate in zone_manager_offload_task:
                zone_manager, candidate_executor = candidate
                # print(f"candidate_executor: {candidate_executor}")
                intersecting_partitions = UtilsFunc().find_line_intersections(
                    (task.creator.x, task.creator.y),
                    (candidate_executor.x, candidate_executor.y),
                    partitions
                )

                if len(intersecting_partitions) > 0 and not isinstance(candidate_executor, CloudNode):
                    attenuation = calcAttenuation(task, candidate_executor, intersecting_partitions)
                    # check : print(blue_bg(f"{attenuation}, {task.creator.x}, {task.creator.y}, {candidate_executor.x}, {candidate_executor.y}"))
                elif isinstance(candidate_executor, CloudNode):
                    attenuation = self.calcAttForCloud(task, intersecting_partitions)
                else:
                    attenuation = 0
                    # locally offloading
                attenuationList.append((zone_manager, candidate_executor, attenuation))
            # logAttenuation(attenuationList)

            # todo: make decision to offload a task
            finalChoiceToOffload, plr = self.noise_controller.makeFinalChoice(attenuationList,
                                                                              task,
                                                                              partitions,
                                                                              Config.NoiseMethod.DEFAULT_METHOD)
            # print(green_bg(f"task.SNR = {task.SNR}"))
            # print(f"plr:{plr}")
            packetLossRandomNumber = random.randint(0, 100)

            # if finalChoiceToOffload:
            #     print(yellow_bg(f"finalChoiceToOffload:{finalChoiceToOffload}"))

            if finalChoiceToOffload:
                chosen_zone_manager, chosen_executor, _ = finalChoiceToOffload

                if packetLossRandomNumber < plr:
                    self.metrics.inc_packet_loss()
                    # print(blue_bg("----------------------------------------------------------------------------"))
                    # print(blue_bg(f"task{task.id}: exec:{chosen_executor.id}, plr:{plr}"))
                    if task in chosen_executor.tasks:
                        chosen_executor.tasks.remove(task)

                    timeout_time = current_time + Config.SimulatorConfig.TIMEOUT_TIME
                    self.schedule_retransmission(task, timeout_time)

                else:
                    self.task_zone_managers[task.id] = chosen_zone_manager
                    # self.metrics.inc_node_tasks(chosen_executor.id)
                    if isinstance(chosen_zone_manager, DeepRLZoneManager):
                        state = chosen_zone_manager.env._get_state(task)  # Get current system state
                        # print(blue_bg(f"------------chosen_executor: {chosen_executor}------------\n------------task: {task}------------"))
                        reward, action = chosen_zone_manager.env._compute_reward2(task, chosen_executor)
                        if not chosen_executor.can_offload_task(task) and (reward > Config.NEGATIVE_REWARD):
                            reward = Config.NEGATIVE_REWARD
                            timeout_time = current_time + 1
                            self.schedule_retransmission(task, timeout_time)
                        elif reward < Config.NEGATIVE_REWARD:
                            timeout_time = current_time + 1
                            self.schedule_retransmission(task, timeout_time)
                        else:
                            chosen_executor.assign_task(task, current_time, self.fixed_fog_nodes)
                            # print(yellow_bg(f"chosen_executor: {chosen_executor.id}"))

                        # if reward < 0:
                        #     print(red_bg(f"reward: --- {reward} --- {task.id}, {chosen_executor.id}"))
                    else:
                        chosen_executor.assign_task(task, current_time, self.fixed_fog_nodes)
                    # if reward < 0:
                    #     print(chosen_executor.remaining_power)
                    if isinstance(chosen_zone_manager, DeepRLZoneManager):
                        next_state = chosen_zone_manager.env._get_state(task)
                        chosen_zone_manager.agent.store_experience(state, action, reward, next_state,
                                                                   done=False)  # Store for training
                        # chosen_zone_manager.agent.train()
            else:
                self.metrics.inc_no_device_found_to_run_becauseOf_Noise()

                timeout_time = current_time + 1
                self.schedule_retransmission(task, timeout_time)
        else:
            if Config.ZoneManagerConfig.DEFAULT_ALGORITHM == Config.ZoneManagerConfig.ALGORITHM_ONLY_FOG:
                self.schedule_retransmission(task, 1)
            elif Config.ZoneManagerConfig.DEFAULT_ALGORITHM == Config.ZoneManagerConfig.ALGORITHM_ONLY_CLOUD:
                self.offload_to_cloud(task, current_time, partitions, self.cloud_node)
            else:
                # print(green_bg("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
                if task.creator.can_offload_task(task):
                    task.creator.assign_task(task, current_time, self.fixed_fog_nodes)
                    # self.metrics.inc_local_execution()
                else:
                    self.offload_to_cloud(task, current_time, partitions, self.cloud_node)

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

    def load_all_predictions_from_csv(self, directory_path: str):
        """
        Scans a directory for prediction CSVs and loads them all into memory.
        This is called once at the start of the simulation.
        """
        print(f"--- Pre-loading all predictions from '{directory_path}' ---")

        # Find all prediction files in the specified directory
        csv_files = glob.glob(os.path.join(directory_path, "prediction_output_*.csv"))

        if not csv_files:
            print(f"Warning: No prediction files found in '{directory_path}'.")
            print("Traffic prediction will be unavailable.")
            return

        total_rows = 0
        for f_path in csv_files:
            try:
                # Read the CSV file
                df = pd.read_csv(f_path)
                # Iterate over its rows and store them in our dictionary
                for row in df.itertuples():
                    # Assumes CSV columns are 'time', 'hex_id', and 'label'
                    self.traffic_predictions[row.time][row.hex_id] = row.label
                    total_rows += 1
            except Exception as e:
                print(f"Error loading prediction file {f_path}: {e}")

        if total_rows > 0:
            min_t = min(self.traffic_predictions.keys())
            max_t = max(self.traffic_predictions.keys())
            print(f"Successfully loaded {total_rows} prediction rows from {len(csv_files)} files.")
            print(f"Pre-loaded data covers timesteps from {min_t} to {max_t}.")
        else:
            print("Warning: No data was loaded from prediction files.")

    def load_cached_traffic(self):
        pkl_dest_dir = Config.Paths.pklPath
        pkl_filename = "traffic_processed_data.pkl"
        full_pkl_path = os.path.join(pkl_dest_dir, pkl_filename)
        pkl_path = full_pkl_path

        print(blue_bg(f"Loading cached traffic data from: {pkl_path}"))

        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, "rb") as f:
                    self.traffic_cache = pickle.load(f)
                print(green_bg("Traffic cache loaded successfully!"))
            except Exception as e:
                print(red_bg(f"Error loading Pickle file: {e}"))
                self.traffic_cache = {}
        else:
            print(red_bg("Cache file not found! Please run the generator script first."))
            self.traffic_cache = {}

    def start_simulation(self):
        self.init_simulation()
        partitions = UtilsFunc.load_partitions("generated_hex_partitions")
        neighbors_map = UtilsFunc.find_neighbors(partitions)

        PREDICTION_UPDATE_INTERVAL = 10
        PREDICTOR_Y_NEEDED = 12

        self.load_cached_traffic()

        while (current_time := self.clock.get_current_time()) < Config.SimulatorConfig.SIMULATION_DURATION:

            print(red_bg(f"current_time:{current_time}"))

            time_int = int(current_time)
            cached_data_str_keys = self.traffic_cache.get(time_int, {})

            traffic_data = {}
            for p in partitions:
                p_name = p.__class__.__name__
                if p_name in cached_data_str_keys:
                    traffic_data[p] = cached_data_str_keys[p_name]
            # print(f"traffic_data:{traffic_data}")

            # --- 2. Store Current Features for Predictor History ---
            print(red_bg(self.traffic_predictions))
            # --- 4. Clean up old predictions ---
            keys_to_delete = [t for t in self.traffic_predictions if t < current_time]
            for t in keys_to_delete: del self.traffic_predictions[t]

            # todo: change this section
            if int(self.clock.get_current_time()) % 5 == 0:
                self.update_rain_with_constraints(neighbors_map, partitions)

            for partition in partitions:
                partition.update_traffic_status(traffic_data)
            # self.logTrafficStatus(partitions)

            soft_tasks = self.load_soft_tasks(current_time)
            self.load_hard_tasks(current_time)
            user_possible_zones = self.assign_mobile_nodes_to_zones(self.user_nodes, layer=Layer.USER)
            mobile_possible_zones = self.assign_mobile_nodes_to_zones(self.mobile_fog_nodes, layer=Layer.FOG)

            merged_possible_zones: Dict[str, List[ZoneManagerABC]] = {**user_possible_zones, **mobile_possible_zones}

            for creator_id, tasks in soft_tasks.items():
                zone_managers = merged_possible_zones.get(creator_id, [])
                self.retransmission(zone_managers, current_time, partitions)

                for task in tasks:
                    self.metrics.inc_total_tasks()
                    zone_manager_offload_task = self.find_zone_manager_offload_task(
                        zone_managers, task, current_time
                    )
                    self.choose_executor_and_assign(
                        zone_manager_offload_task, task, partitions, current_time
                    )

            target_id = "PKW105"
            self.print_node_schedule_status(current_time, target_id)

            self.update_graph()
            self.execute_tasks_for_one_step()
            self.metrics.flush()

            self.metrics.log_metrics()
            self.metrics.add_data(current_time)

            # if current_time % 50 == 0:
            #     self.metrics.print_node_tasks()
            # if current_time == 500:
            #     self.metrics.saveToExcel("test.csv")

        self.drop_not_completed_tasks()
        self.save_missed_deadlines_to_excel(
            f"missed_deadlines_report_{Config.ZoneManagerConfig.DEFAULT_ALGORITHM}_{Config.NoiseMethod.DEFAULT_METHOD}_{Config.NoiseConfig.DEFAULT_THRESHOLD}_{Config.TrafficNoise.DEFAULT_TrafficNoiseLevel}_{Config.AttenuationLevel.DEFAULT_AttenuationLevelName}_{Config.City.DEFAULT_CITY}.xlsx")
        self.save_success_deadlines_to_excel(
            f"success_deadlines_report_{Config.ZoneManagerConfig.DEFAULT_ALGORITHM}_{Config.NoiseMethod.DEFAULT_METHOD}_{Config.NoiseConfig.DEFAULT_THRESHOLD}_{Config.TrafficNoise.DEFAULT_TrafficNoiseLevel}_{Config.AttenuationLevel.DEFAULT_AttenuationLevelName}_{Config.City.DEFAULT_CITY}.xlsx")
        self.metrics.save_to_excel(
            f"final_metrics_summary_{Config.ZoneManagerConfig.DEFAULT_ALGORITHM}_{Config.NoiseMethod.DEFAULT_METHOD}_{Config.NoiseConfig.DEFAULT_THRESHOLD}_{Config.TrafficNoise.DEFAULT_TrafficNoiseLevel}_{Config.AttenuationLevel.DEFAULT_AttenuationLevelName}_{Config.City.DEFAULT_CITY}.xlsx")

    def _resolve_task_creator(self, creator_id: str) -> Optional[MobileNodeABC]:
        if creator_id in self.user_nodes:
            return self.user_nodes[creator_id]
        if creator_id in self.mobile_fog_nodes:
            return self.mobile_fog_nodes[creator_id]
        return None

    def load_soft_tasks(self, current_time: float) -> Dict[str, List[Task]]:
        """Load soft tasks and attach their creators for the offload path."""
        tasks: Dict[str, List[Task]] = defaultdict(list)
        for creator_id, creator_tasks in self.loader.load_nodes_tasks(current_time).items():
            creator = self._resolve_task_creator(creator_id)
            if creator is None:
                print(f"there is no creator for soft task: {creator_id}\n")
                continue
            for task in creator_tasks:
                task.creator = creator
                tasks[creator_id].append(task)
        return tasks

    def load_hard_tasks(self, current_time: float) -> int:
        """Load hard tasks into each vehicle's critical processor for local EDF scheduling."""
        loaded_count = 0
        for creator_id, creator_tasks in self.loader.load_nodes_hard_tasks(current_time).items():
            creator = self._resolve_task_creator(creator_id)
            if creator is None:
                print(f"there is no creator for hard task: {creator_id}\n")
                continue
            for task in creator_tasks:
                self._assign_hard_task_to_critical_processor(task, creator, current_time)
                self.metrics.inc_total_tasks()
                loaded_count += 1
        return loaded_count

    @staticmethod
    def _assign_hard_task_to_critical_processor(
            task: Task,
            creator: MobileNodeABC,
            current_time: float,
    ) -> None:
        """Enqueue a pre-generated periodic task on the vehicle's critical processor."""
        critical = creator.critical_processor
        task.creator = creator
        task.creator_id = f"#{critical.id}"
        task.release_time = current_time
        task.remaining_time = task.exec_time
        task.start_time = current_time
        task.is_hard = True
        critical.periodic_jobs_active.append(task)

    def load_tasks(self, current_time: float) -> Dict[str, List[Task]]:
        """Load soft tasks; hard tasks are loaded via load_hard_tasks()."""
        self.load_hard_tasks(current_time)
        return self.load_soft_tasks(current_time)

    def execute_tasks_for_one_step(self):
        executed_tasks: List[Task] = []
        merged_nodes: Dict[str, NodeABC] = {
            **self.mobile_fog_nodes,
            **self.user_nodes,
            **self.fixed_fog_nodes,
            self.cloud_node.id: self.cloud_node,
        }
        for node_id, node in merged_nodes.items():
            tasks = node.execute_tasks(self.clock.get_current_time(), self.fixed_fog_nodes)
            # if tasks:
            #     print(yellow_bg(f"tasks :{tasks}"))
            executed_tasks.extend(tasks)
            for task in tasks:
                zone_manager = self.task_zone_managers.get(task.id)
                if zone_manager:
                    zone_manager.update(current_task=task)
                    all_fog_nodes = {**zone_manager.fixed_fog_nodes, **zone_manager.mobile_fog_nodes}
                    loads = [len(node.tasks) for node in all_fog_nodes.values() if node.can_offload_task(task)]
                    if loads:
                        min_load = min(loads)
                        max_load = max(loads)
                        self.metrics.inc_task_load_diff(task.id, min_load, max_load)
                if task.is_hard:
                    self.metrics.inc_local_execution()
                elif isinstance(task.executor, (FixedFogNode, MobileFogNode)):
                    self.metrics.inc_fog_execution()
                elif task.creator.id == task.executor.id:
                    self.metrics.inc_local_execution()
                elif isinstance(task.executor, CloudNode):
                    self.metrics.inc_cloud_tasks()
                # if task.has_migrated:
                #     self.metrics.inc_migration()
                # if task.has_migrated and task.is_deadline_missed:
                #     self.metrics.inc_migrate_and_miss()
                if task.is_deadline_missed:
                    # print(blue_bg(
                    #     f"{task.id}: release_time:{task.release_time}, deadline:{task.deadline}, exec_time:{task.exec_time}, finish_time:{task.finish_time}, {task.executor.id}, {task.dataSize}, diff:{task.finish_time-task.deadline}"))
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

    def offload_to_cloud(self, task: Task, current_time: float, partitions, cloud_node):
        if self.cloud_node.can_offload_task(task):
            # todo: complete this fucking shit
            attenuationList = []
            intersecting_partitions = UtilsFunc().find_line_intersections(
                (task.creator.x, task.creator.y),
                (self.cloud_node.x, self.cloud_node.y),
                partitions
            )
            attenuation = self.calcAttForCloud(task, intersecting_partitions)
            attenuationList.append((None, cloud_node, attenuation))
            finalChoiceToOffload, plr = self.noise_controller.makeFinalChoice(attenuationList,
                                                                              task,
                                                                              partitions,
                                                                              Config.NoiseMethod.DEFAULT_METHOD)

            # print(f"plr:{plr}")
            packetLossRandomNumber = random.randint(0, 100)

            if finalChoiceToOffload:
                chosen_zone_manager, chosen_executor, _ = finalChoiceToOffload

                if packetLossRandomNumber < plr:
                    self.metrics.inc_packet_loss()
                    if task in chosen_executor.tasks:
                        chosen_executor.tasks.remove(task)

                    timeout_time = current_time + Config.SimulatorConfig.TIMEOUT_TIME
                    self.schedule_retransmission(task, timeout_time)

                else:
                    self.task_zone_managers[task.id] = chosen_zone_manager
                    # self.metrics.inc_node_tasks(chosen_executor.id)

                    self.cloud_node.assign_task(task, current_time, self.fixed_fog_nodes)

            else:
                self.metrics.inc_no_device_found_to_run_becauseOf_Noise()

                timeout_time = current_time + 1
                self.schedule_retransmission(task, timeout_time)

        else:
            self.schedule_retransmission(task, 1)

    def assign_mobile_nodes_to_zones(
            self,
            mobile_nodes: dict[str, MobileNodeABC],
            layer: Layer
    ) -> Dict[str, List[ZoneManagerABC]]:

        nodes_possible_zones: Dict[str, List[ZoneManagerABC]] = defaultdict(list)
        for z_id, zone_manager in self.zone_managers.items():
            nodes: List[MobileNodeABC] = []
            for n_id, mobile_node in mobile_nodes.items():
                if zone_manager.zone.is_in_coverage(mobile_node.x, mobile_node.y):
                    nodes.append(mobile_node)
                    nodes_possible_zones[n_id].append(zone_manager)
            if layer == Layer.FOG:
                zone_manager.set_mobile_fog_nodes(nodes)
        return nodes_possible_zones

    def update_mobile_fog_nodes_coordinate(self) -> None:
        new_nodes_data = self.loader.load_mobile_fog_nodes(self.clock.get_current_time())
        self.mobile_fog_nodes = self.update_nodes_coordinate(self.mobile_fog_nodes, new_nodes_data)

    def update_user_nodes_coordinate(self) -> None:
        new_nodes_data = self.loader.load_user_nodes(self.clock.get_current_time())
        self.user_nodes = self.update_nodes_coordinate(self.user_nodes, new_nodes_data)

    @staticmethod
    def update_nodes_coordinate(old_nodes: dict[str, MobileNodeABC], new_nodes: dict[str, MobileNodeABC]):
        data: Dict[str, MobileNodeABC] = {}
        for n_id, new_node in new_nodes.items():
            if n_id not in old_nodes:
                node = new_node
            else:
                node = old_nodes[n_id]
                node.x = new_node.x
                node.y = new_node.y
                node.angle = new_node.angle
                node.speed = new_node.speed
                if hasattr(node, "critical_processor"):
                    node.critical_processor.x = new_node.x
                    node.critical_processor.y = new_node.y
            data[n_id] = node
        return data

    def drop_not_completed_tasks(self) -> List[Task]:
        left_tasks: list[Task] = []
        merged_nodes: Dict[str, NodeABC] = {
            **self.mobile_fog_nodes,
            **self.user_nodes,
            self.cloud_node.id: self.cloud_node,
        }

        for node_id, node in merged_nodes.items():
            left_tasks.extend(node.tasks)
            for _ in range(len(node.tasks)):
                self.metrics.inc_deadline_miss()
            if hasattr(node, "critical_processor"):
                critical = node.critical_processor
                left_tasks.extend(critical.periodic_jobs_active)
                left_tasks.extend(critical.tasks)
                for _ in range(len(critical.periodic_jobs_active) + len(critical.tasks)):
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

    def print_node_schedule_status(self, current_time: float, target_node_id: str):
        """Prints the scheduling status and core loads of a specific node."""
        merged_nodes = {
            **self.mobile_fog_nodes,
            **self.user_nodes,
            **self.fixed_fog_nodes,
            self.cloud_node.id: self.cloud_node,
        }

        node = merged_nodes.get(target_node_id)
        if not node:
            return

        print(f"\n{'=' * 65}")
        print(f"🕒 Time Step: {current_time:.2f} | 🚗 Node ID: {target_node_id}")
        print(f"{'=' * 65}")

        if not hasattr(node, 'num_cores') or not hasattr(node, 'cores'):
            print("⚠️ This node does not support multi-core scheduling!")
            return

        print("🖥️  Core Scheduling Status:")
        for i in range(node.num_cores):
            core_heap = node.cores[i]
            load = node.core_loads[i]

            bar_length = min(int(load * 2), 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)

            print(f"\n  [Core {i}] Load: {load:.2f} |{bar}|")

            if not core_heap:
                print("      └_ 💤 Core is idle (No tasks in queue).")
            else:
                sorted_tasks = sorted(core_heap, key=lambda x: x[0])
                for idx, (deadline, rel_time, task) in enumerate(sorted_tasks):
                    print(f"      ├_ Task ID: {task.id} (Creator: {task.creator_id})")
                    progress = 0
                    if hasattr(task, 'total_exec_time') and task.total_exec_time > 0:
                        progress = ((task.total_exec_time - task.remaining_time) / task.total_exec_time) * 100

                    print(
                        f"      │  ├_ Exec Time: {task.remaining_time:.2f}s remaining of {task.total_exec_time:.2f}s ({progress:.1f}% done)")
                    print(f"      │  ├_ Release Time: {task.release_time:.2f}")
                    print(f"      │  ├_ Deadline: {task.deadline:.2f}")
                    print(f"      │  └_ Assigned Start Time: {task.start_time:.2f}")

        print(f"{'=' * 65}\n")
