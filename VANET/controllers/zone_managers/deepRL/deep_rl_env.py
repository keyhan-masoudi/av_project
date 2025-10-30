import csv
import math

import numpy as np
import gym
from gym import spaces
from config import Config
from models.node.base import findExecTimeInEachKindOfNode, find_closest_fn, findDataRate
from models.node.fog import FixedFogNode, MobileFogNode
from utils.distance import get_distance


def red_bg(text):
    return f"\033[41m{text}\033[0m"

def purple_bg(text):
    return f"\033[45m{text}\033[0m"

def get_vehicle_position(csv_file, target_id):
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['vehicle_id'] == target_id:
                x = float(row['x'])
                y = float(row['y'])
                return x, y
    return None, None


def checkMigration(executor, task, finishTime):
    finishTime = math.floor(finishTime)
    if finishTime > 1200:
        return False
    fileName = f"E:\pythonProject\VANET\SumoDividedByTime\Outputs2\dataInTime{int(finishTime)}.csv"
    creatorX, creatorY = get_vehicle_position(fileName, task.creator_id)
    if (creatorX is None) or (creatorY is None):
        return True
    if executor.radius > np.sqrt((creatorX - executor.x) ** 2 + (creatorY - executor.y) ** 2):
        return False
    return True


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def isDeadlineMissHappening(task, executor, fn_nodes):
    task.real_exec_time_base = findExecTimeInEachKindOfNode(task, executor)

    real_exec_time = task.real_exec_time_base

    if executor == task.creator:
        return ((task.release_time + real_exec_time) > task.deadline), (
                task.deadline - (task.release_time + real_exec_time))
    elif isinstance(executor, (FixedFogNode, MobileFogNode)):
        # if checkMigration(executor, task, (task.release_time + real_exec_time)):
        #     real_exec_time += Config.TaskConfig.MIGRATION_OVERHEAD * task.dataSize
        dataRate = findDataRate(task, executor, 0)
        # print(purple_bg(f"{executor.id} ===> dataRate : {dataRate}, task.dataSize: {task.dataSize} ===> transmission time :{task.dataSize / dataRate}"))
        real_exec_time += task.dataSize / dataRate

        return ((task.release_time + real_exec_time) > task.deadline), (
                task.deadline - (task.release_time + real_exec_time))
    else:
        closest_fn = find_closest_fn(task.creator.x, task.creator.y, fn_nodes, task.power)
        dataRate = findDataRate(task, executor, closest_fn)
        # print(f"closest_fn:{closest_fn}, x: {closest_fn}")
        if closest_fn.x == 4214.90 and closest_fn.y == 1932.26:
            real_exec_time += (task.dataSize / dataRate) + (
                    task.dataSize / Config.CloudConfig.CLOUD_BANDWIDTH)
        else:
            real_exec_time += (task.dataSize / dataRate) + 2 * (
                    task.dataSize / Config.CloudConfig.CLOUD_BANDWIDTH)
        # print(purple_bg(f"{executor.id} ===> dataRate : {dataRate}, task.dataSize: {task.dataSize} ===> transmission time :{(task.dataSize / dataRate) + 2 * (task.dataSize / Config.CloudConfig.CLOUD_BANDWIDTH)}"))

        # real_exec_time += Config.TaskConfig.CLOUD_PROCESSING_OVERHEAD
        return ((task.release_time + real_exec_time) > task.deadline), (
                task.deadline - (task.release_time + real_exec_time))


class DeepRLEnvironment(gym.Env):
    """
    Custom environment for RL-based task offloading.
    """

    def __init__(self, simulator):
        # print(f"test2:{simulator}")
        super(DeepRLEnvironment, self).__init__()
        # print(f"test3:{simulator}")

        self.simulator = simulator  # Reference to the existing simulation
        self.metrics = simulator.metrics  # Track performance

        # Define action space: (Where to offload the task?)
        self.action_space = spaces.Discrete(3)  # 0: Local, 1: Fog, 2: Cloud

        # Define state space: (What information do we use to make decisions?)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5,), dtype=np.float32
        )

    def reset(self):
        """Reset the environment to start a new episode."""
        # print(f"test4:{self.simulator}")
        self.simulator.init_simulation()
        return self._get_state()

    def step(self, action):
        """Execute an action and return the next state, reward, and done flag."""
        task = self.simulator.get_next_task()  # Get the next task to process

        if task is None:
            # print(red_bg("++++++++++++++++++++++++++++"))
            self.simulator.update_graph()
            done = self.simulator.clock.get_current_time() >= \
                   Config.SimulatorConfig.SIMULATION_DURATION
            return self._get_state(), 0, done, {}  # No task left, episode ends

        reward = self._execute_action(task, action)  # Execute offloading
        print(red_bg(f"reward: {reward}"))
        next_state = self._get_state()
        done = self.simulator.clock.get_current_time() >= Config.SimulatorConfig.SIMULATION_DURATION

        return next_state, reward, done, {}

    def _execute_action(self, task, action):
        """Perform the task offloading based on the action and return the reward."""
        if action == 0:
            candidate_executor = task.creator  # Local execution
        elif action == 1:
            candidate_executor = self._get_best_fog_node(task)  # Offload to fog
        else:
            candidate_executor = self.simulator.cloud_node  # Offload to cloud

        if candidate_executor and candidate_executor.can_offload_task(task):
            # executor.assign_task(task, self.simulator.clock.get_current_time())  # note : i have removed this line to have multi agent algorithm
            reward = self._compute_reward2(task, candidate_executor)
        else:
            reward = -1  # Task couldn't be offloaded

        return reward

    def _get_best_fog_node(self, task):
        creator = task.creator
        all_fog_nodes = list(self.fixed_fog_nodes.values()) + list(self.mobile_fog_nodes.values())
        eligible_nodes = [node for node in all_fog_nodes if node.can_offload_task(task)]

        if not eligible_nodes:
            return None

        if len(eligible_nodes) == 1:
            return eligible_nodes[0]

        distances_by_id = {
            node.id: get_distance(node.x, node.y, creator.x, creator.y)
            for node in eligible_nodes
        }

        min_dist = min(distances_by_id.values())
        max_dist = max(distances_by_id.values())

        mobile_node_ids = {node.id for node in self.mobile_fog_nodes.values()}

        def calculate_score(node):
            if node.id in mobile_node_ids:
                normalized_power = node.power / Config.MobileFogNodeConfig.DEFAULT_COMPUTATION_POWER
            else:
                normalized_power = node.power / Config.FixedFogNodeConfig.DEFAULT_COMPUTATION_POWER

            distance = distances_by_id[node.id]

            if max_dist == min_dist:
                normalized_distance = 0.0
            else:
                normalized_distance = (distance - min_dist) / (max_dist - min_dist)

            distance_score = 1.0 - normalized_distance

            final_score = (0.5 * normalized_power) + (0.5 * distance_score)
            return final_score

        chosen_node = max(
            eligible_nodes,
            key=calculate_score,
            default=None
        )

        return chosen_node

    def _compute_reward(self, task, executor):
        """Compute the reward based on execution success, latency, and power efficiency."""
        if executor == task.creator:
            return 1.0  # Local execution is preferred (low cost)
        elif isinstance(executor, (FixedFogNode, MobileFogNode)):
            return 2.0  # Fog execution is better than cloud
        else:
            return 0.5  # Cloud execution has higher cost

    def _compute_reward2(self, task, executor):
        """Compute the reward based on latency."""
        # todo: should add execTime and check deadline

        """
        Reward function based on task completion timing.
        If task is late (lateness < 0): reward = -2 + lateness
        If task is on-time or early: reward = lateness
        """
        reward = 0
        isDeadlineMiss, lateness = isDeadlineMissHappening(task, executor, self.simulator.fixed_fog_nodes)
        # print(green_bg(f"{executor.id}: {lateness}"))
        if isDeadlineMiss:
            # print(f"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA:{task.id}: {executor.id}")
            reward = -100 + lateness
            # print(red_bg(f"{reward}, {lateness}"))
        else:
            reward = lateness
            # print(red_bg(f"{reward}, {lateness}"))
        if executor == task.creator:
            return reward, 0
        elif isinstance(executor, (FixedFogNode, MobileFogNode)):
            return reward, 1
        else:
            return reward, 2

    def _calculate_avg_fog_power(self, vehicle):
        """
        Calculates the average remaining power of all fog nodes
        within 300 meters of the given vehicle.
        """
        fog_nodes = list(self.simulator.mobile_fog_nodes.values()) + list(self.simulator.fixed_fog_nodes.values())

        # Filter fog nodes within 300 meters of the vehicle
        nearby_fogs = []
        for fog in fog_nodes:
            distance = np.sqrt((fog.x - vehicle.x) ** 2 + (fog.y - vehicle.y) ** 2)
            if distance <= 300:
                nearby_fogs.append(fog)

        if len(nearby_fogs) == 0:
            return 0.0

        avg_power = sum(node.remaining_power for node in nearby_fogs) / len(nearby_fogs)
        return avg_power

    def _calculate_max_fog_power(self, vehicle):
        """
        Calculates the maximum remaining power of all fog nodes
        within 300 meters of the given vehicle.
        """
        fog_nodes = list(self.simulator.mobile_fog_nodes.values()) + list(self.simulator.fixed_fog_nodes.values())

        # Filter fog nodes within 300 meters of the vehicle
        nearby_fogs = []
        for fog in fog_nodes:
            distance = np.sqrt((fog.x - vehicle.x) ** 2 + (fog.y - vehicle.y) ** 2)
            if distance <= 300:
                nearby_fogs.append(fog)

        if len(nearby_fogs) == 0:
            return 0.0

        max_power = max(node.remaining_power for node in nearby_fogs)
        return max_power

    def _get_state(self, task=None):
        """
        Extract the state vector for the RL agent.
        State format:
        [remainingVehiclePower, taskPower, timeToExecute, avg_fog_available_power, vehicleSpeed, cloud_available_power]
        """
        if task is not None:
            remaining_power = task.creator.remaining_power if task.creator else 0.0
            task_power = task.power
            vehicle_speed = task.creator.speed if hasattr(task.creator, 'speed') else 0.0
            time_to_execute = task.exec_time  # in normal mode
            # note: maybe it's needed to add /2 for fog and cloud, but how?? (i think it's okay now and it's considered in reward)

        else:
            remaining_power = 0.0
            task_power = 0.0
            time_to_execute = 0.0
            vehicle_speed = 0.0

        # exec time ratio
        maxExecTime = 25.0
        execTimeRatio = time_to_execute / maxExecTime

        # task power ratio
        maxTaskPower = 3.5
        taskPowerRatio = task_power / maxTaskPower

        # vehicle speed ratio
        # maxSpeedOfaVehicle = 13.89
        # vehicle_speed_ratio = vehicle_speed / maxSpeedOfaVehicle

        # vehicle remaining power ratio
        maxVehiclePower = task.creator.power
        VehiclePowerRatio = remaining_power / maxVehiclePower

        # fog power ratio
        avg_fog_power = self._calculate_avg_fog_power(task.creator)
        max_fog_power_in_range = self._calculate_max_fog_power(task.creator)
        max_fog_power = 19.79
        avg_fog_remaining_power_ratio = avg_fog_power / max_fog_power
        max_fog_remaining_power_ratio = max_fog_power_in_range / max_fog_power

        # cloud power ratio
        cloud_remaining_power = self.simulator.cloud_node.remaining_power if self.simulator.cloud_node else 0.0
        cloud_power = self.simulator.cloud_node.power if self.simulator.cloud_node else 1.0
        cloud_power_ratio = cloud_remaining_power / cloud_power

        return np.array([
            VehiclePowerRatio,
            taskPowerRatio,
            execTimeRatio,
            avg_fog_remaining_power_ratio,
            # max_fog_remaining_power_ratio,
            # vehicle_speed_ratio,
            cloud_power_ratio
        ], dtype=np.float32)
