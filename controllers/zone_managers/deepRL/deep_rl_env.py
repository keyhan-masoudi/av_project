import csv
import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from config import Config
from models.node.cloud import CloudNode
from task_and_user_generator import Config as CNF
from models.node.base import findExecTimeInEachKindOfNode, find_closest_fn, findDataRate
from models.node.fog import FixedFogNode, MobileFogNode
from utils.distance import get_distance
from collections import deque


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


# def checkMigration(executor, task, finishTime):
#     finishTime = math.floor(finishTime)
#     if finishTime > 1200:
#         return False
#     fileName = f"E:\pythonProject\VANET\SumoDividedByTime\Outputs2\dataInTime{int(finishTime)}.csv"
#     creatorX, creatorY = get_vehicle_position(fileName, task.creator_id)
#     if (creatorX is None) or (creatorY is None):
#         return True
#     if executor.radius > np.sqrt((creatorX - executor.x) ** 2 + (creatorY - executor.y) ** 2):
#         return False
#     return True


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# todo: should change this
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
        if closest_fn.x == Config.CloudConfig.CLOSEST_FOG_X and closest_fn.y == Config.CloudConfig.CLOSEST_FOG_Y:
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
        # self.action_space = spaces.Discrete(3)  # 0: Local, 1: Fog, 2: Cloud
        # 0:Local, 1:Cloud, 2:Fog1, 3:Fog2, 4:Fog3
        self.action_space = spaces.Discrete(5)

        # Define state space: (What information do we use to make decisions?)
        # self.observation_space = spaces.Box(
        #     low=0, high=1, shape=(5,), dtype=np.float32
        # )
        # 36: Task(3) + Local(5) + Env(7) + Fog(15) + Cloud(6)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(36,), dtype=np.float32
        )

        # Frame Stacking for weather
        self.weather_history = deque([0.0, 0.0, 0.0], maxlen=3)

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

    # def get_action_mask(self, task):
    #     mask = [1.0, 1.0, 1.0]
    #     if task is not None:
    #         local_exec_time = findExecTimeInEachKindOfNode(task, task.creator)
    #
    #         if (task.release_time + local_exec_time) > task.deadline:
    #             mask[0] = 0.0
    #
    #     return np.array(mask, dtype=np.float32)

    def get_action_mask(self, task):
        # [Local, Cloud, Fog1, Fog2, Fog3]
        mask = np.ones(5, dtype=np.float32)

        creator = task.creator

        # todo: change this variables
        # todo: MAX_QUEUE_LEN should not be a constant variable
        # todo: maybe it would be a better option if i use can_offload instead of this
        MAX_QUEUE_LEN = 20.0
        MAX_DISTANCE = 500.0  # حداکثر شعاع ارتباطی ماشین

        # =====================================
        # 1. مسک کردن Local (اکشن 0)
        # =====================================
        local_best_q = getattr(creator, 'get_best_queue_length', lambda: 0.0)()
        # اگر صف خلوت‌ترین کور از ۸۵ درصد ظرفیت بیشتر بود، لوکال را مسک کن
        if (local_best_q / MAX_QUEUE_LEN) > 0.85:
            mask[0] = 0.0

        # todo: maybe it won't be bad if we mask fog with queue too
        # =====================================
        # 3. مسک کردن Fog ها (اکشن‌های 2, 3, 4)
        # =====================================
        nearest_fogs = self._get_k_nearest_fogs(creator, k=3)

        for i in range(3):
            action_idx = i + 2  # نگاشت i=0 به اکشن 2 و الی آخر

            if i < len(nearest_fogs):
                fog = nearest_fogs[i]
                dist = np.sqrt((fog.x - creator.x) ** 2 + (fog.y - creator.y) ** 2)

                if dist > MAX_DISTANCE:
                    mask[action_idx] = 0.0
            else:
                mask[action_idx] = 0.0

        return mask

    # def _execute_action(self, task, action):
    #     """Perform the task offloading based on the action and return the reward."""
    #     if action == 0:
    #         candidate_executor = task.creator  # Local execution
    #     elif action == 1:
    #         candidate_executor = self._get_best_fog_node(task)  # Offload to fog
    #     else:
    #         candidate_executor = self.simulator.cloud_node  # Offload to cloud
    #
    #     if candidate_executor and candidate_executor.can_offload_task(task):
    #         # executor.assign_task(task, self.simulator.clock.get_current_time())  # note : i have removed this line to have multi agent algorithm
    #         reward = self._compute_reward2(task, candidate_executor)
    #     else:
    #         reward = -1  # Task couldn't be offloaded
    #
    #     return reward

    # def _get_best_fog_node(self, task):
    #     creator = task.creator
    #     all_fog_nodes = list(self.fixed_fog_nodes.values()) + list(self.mobile_fog_nodes.values())
    #     eligible_nodes = [node for node in all_fog_nodes if node.can_offload_task(task)]
    #
    #     if not eligible_nodes:
    #         return None
    #
    #     if len(eligible_nodes) == 1:
    #         return eligible_nodes[0]
    #
    #     distances_by_id = {
    #         node.id: get_distance(node.x, node.y, creator.x, creator.y)
    #         for node in eligible_nodes
    #     }
    #
    #     min_dist = min(distances_by_id.values())
    #     max_dist = max(distances_by_id.values())
    #
    #     mobile_node_ids = {node.id for node in self.mobile_fog_nodes.values()}
    #
    #     def calculate_score(node):
    #         if node.id in mobile_node_ids:
    #             normalized_power = node.power / Config.MobileFogNodeConfig.DEFAULT_COMPUTATION_POWER
    #         else:
    #             normalized_power = node.power / Config.FixedFogNodeConfig.DEFAULT_COMPUTATION_POWER
    #
    #         distance = distances_by_id[node.id]
    #
    #         if max_dist == min_dist:
    #             normalized_distance = 0.0
    #         else:
    #             normalized_distance = (distance - min_dist) / (max_dist - min_dist)
    #
    #         distance_score = 1.0 - normalized_distance
    #
    #         final_score = (0.5 * normalized_power) + (0.5 * distance_score)
    #         return final_score
    #
    #     chosen_node = max(
    #         eligible_nodes,
    #         key=calculate_score,
    #         default=None
    #     )
    #
    #     return chosen_node

    def _execute_action(self, task, action):
        """Perform the task offloading based on the action and return the reward."""
        candidate_executor = None

        if action == 0:
            candidate_executor = task.creator

        elif action == 1:
            candidate_executor = self.simulator.cloud_node

        elif action in [2, 3, 4]:
            fog_index = action - 2
            nearest_fogs = self._get_k_nearest_fogs(task.creator, k=3)

            if fog_index < len(nearest_fogs):
                candidate_executor = nearest_fogs[fog_index]

        # If the agent randomly selects a non-existent fog in Exploration mode
        if candidate_executor is None:
            return -1.0

        # todo: should write new reward formulation
        if candidate_executor.can_offload_task(task):
            reward = self._compute_reward_complex(task, candidate_executor)
        else:
            reward = -10.0
        return reward

    def _get_k_nearest_fogs(self, vehicle, k=3):
        all_fogs = list(self.simulator.fixed_fog_nodes.values()) + list(self.simulator.mobile_fog_nodes.values())
        all_fogs.sort(key=lambda fog: np.sqrt((fog.x - vehicle.x) ** 2 + (fog.y - vehicle.y) ** 2))
        return all_fogs[:k]

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

    # def _calculate_avg_fog_power(self, vehicle):
    #     """
    #     Calculates the average remaining power of all fog nodes
    #     within 300 meters of the given vehicle.
    #     """
    #     fog_nodes = list(self.simulator.mobile_fog_nodes.values()) + list(self.simulator.fixed_fog_nodes.values())
    #
    #     # Filter fog nodes within 300 meters of the vehicle
    #     nearby_fogs = []
    #     for fog in fog_nodes:
    #         distance = np.sqrt((fog.x - vehicle.x) ** 2 + (fog.y - vehicle.y) ** 2)
    #         if distance <= 300:
    #             nearby_fogs.append(fog)
    #
    #     if len(nearby_fogs) == 0:
    #         return 0.0
    #
    #     avg_power = sum(node.remaining_power for node in nearby_fogs) / len(nearby_fogs)
    #     return avg_power

    # def _calculate_max_fog_power(self, vehicle):
    #     """
    #     Calculates the maximum remaining power of all fog nodes
    #     within 300 meters of the given vehicle.
    #     """
    #     fog_nodes = list(self.simulator.mobile_fog_nodes.values()) + list(self.simulator.fixed_fog_nodes.values())
    #
    #     # Filter fog nodes within 300 meters of the vehicle
    #     nearby_fogs = []
    #     for fog in fog_nodes:
    #         distance = np.sqrt((fog.x - vehicle.x) ** 2 + (fog.y - vehicle.y) ** 2)
    #         if distance <= 300:
    #             nearby_fogs.append(fog)
    #
    #     if len(nearby_fogs) == 0:
    #         return 0.0
    #
    #     max_power = max(node.remaining_power for node in nearby_fogs)
    #     return max_power

    # def _get_state(self, task=None):
    #     """
    #     Extract the state vector for the RL agent.
    #     State format:
    #     [remainingVehiclePower, taskPower, timeToExecute, avg_fog_available_power, vehicleSpeed, cloud_available_power]
    #     """
    #     if task is not None:
    #         remaining_power = task.creator.remaining_power if task.creator else 0.0
    #         task_power = task.power
    #         # vehicle_speed = task.creator.speed if hasattr(task.creator, 'speed') else 0.0
    #         time_to_execute = task.exec_time  # in normal mode
    #         # note: maybe it's needed to add /2 for fog and cloud, but how?? (i think it's okay now and it's considered in reward)
    #
    #     else:
    #         remaining_power = 0.0
    #         task_power = 0.0
    #         time_to_execute = 0.0
    #         # vehicle_speed = 0.0
    #
    #     # exec time ratio
    #     maxExecTime = 25.0
    #     execTimeRatio = time_to_execute / maxExecTime
    #
    #     # task power ratio
    #     maxTaskPower = 3.5
    #     taskPowerRatio = task_power / maxTaskPower
    #
    #     # vehicle speed ratio
    #     # maxSpeedOfaVehicle = 13.89
    #     # vehicle_speed_ratio = vehicle_speed / maxSpeedOfaVehicle
    #
    #     # vehicle remaining power ratio
    #     maxVehiclePower = task.creator.power
    #     VehiclePowerRatio = remaining_power / maxVehiclePower
    #
    #     # fog power ratio
    #     avg_fog_power = self._calculate_avg_fog_power(task.creator)
    #     max_fog_power_in_range = self._calculate_max_fog_power(task.creator)
    #     max_fog_power = 19.79
    #     avg_fog_remaining_power_ratio = avg_fog_power / max_fog_power
    #     max_fog_remaining_power_ratio = max_fog_power_in_range / max_fog_power
    #
    #     # cloud power ratio
    #     cloud_remaining_power = self.simulator.cloud_node.remaining_power if self.simulator.cloud_node else 0.0
    #     cloud_power = self.simulator.cloud_node.power if self.simulator.cloud_node else 1.0
    #     cloud_power_ratio = cloud_remaining_power / cloud_power
    #
    #     return np.array([
    #         VehiclePowerRatio,
    #         taskPowerRatio,
    #         execTimeRatio,
    #         avg_fog_remaining_power_ratio,
    #         # max_fog_remaining_power_ratio,
    #         # vehicle_speed_ratio,
    #         cloud_power_ratio
    #     ], dtype=np.float32)

    def _get_state(self, task=None, current_time=None):
        if task is None:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        state_vector = []
        creator = task.creator
        executor = task.executor

        # === ثوابت نرمال‌سازی (باید در فایل Config قرار بگیرند) ===
        MAX_CAPACITY = 20.0  # حداکثر ظرفیت یک گره فاگ/لوکال
        MAX_CLOUD_CAPACITY = 100.0  # حداکثر ظرفیت کلاد
        MAX_QUEUE_LEN = 20.0  # حداکثر طول صف مجاز
        MAX_CORES = 8.0  # حداکثر تعداد کورها
        MAX_DISTANCE = 500.0  # شعاع ارتباطی محیط
        MAX_BANDWIDTH = 100.0  # حداکثر پهنای باند کلاد
        MAX_DELAY = 2.0  # حداکثر تاخیر پایه کلاد

        # ==========================================
        # بلوک ۱: ویژگی‌های Task
        # ==========================================
        data_size_ratio = task.dataSize / CNF.TaskConfig.MAX_DATASIZE
        workload_ratio = task.power / CNF.TaskConfig.MAX_POWER_CONSUMPTION
        # todo: i have recently add current time, so i should validate this parameter here
        deadline_ratio = (task.deadline - current_time) / CNF.TaskConfig.DEADLINE_MAX_FREE_TIME

        state_vector.extend([data_size_ratio, workload_ratio, deadline_ratio])

        # ==========================================
        # بلوک ۲: ویژگی‌های Local Node
        # ==========================================
        # todo: maybe i should change power values or remove this section
        local_tot_cap = creator.power / Config.FixedFogNodeConfig.DEFAULT_COMPUTATION_POWER
        local_rem_cap = creator.remaining_power / Config.FixedFogNodeConfig.DEFAULT_COMPUTATION_POWER

        # این متدها باید به کلاس پایه گره‌ها (Node) اضافه شوند
        # todo: should change MAX_QUEUE_LEN to sth which i don't know now
        local_best_q = creator.get_best_queue_length() / MAX_QUEUE_LEN
        local_avg_q = creator.get_avg_queue_length() / MAX_QUEUE_LEN
        local_idle_cores = -1
        if isinstance(executor, FixedFogNode):
            local_idle_cores = executor.get_idle_cores_count() / Config.FixedFogNodeConfig.NUM_CORE
        elif isinstance(executor, MobileFogNode):
            local_idle_cores = executor.get_idle_cores_count() / Config.MobileFogNodeConfig.NUM_CORE
        elif task.creator.id == executor.id:
            local_idle_cores = executor.get_idle_cores_count() / Config.UserNodeConfig.NUM_CORE
        elif isinstance(executor, CloudNode):
            local_idle_cores = executor.get_idle_cores_count() / Config.CloudConfig.NUM_CORE

        state_vector.extend([local_tot_cap, local_rem_cap, local_best_q, local_avg_q, local_idle_cores])

        # ==========================================
        # بلوک ۳: ویژگی‌های محیط (Environment & Context)
        # ==========================================
        current_traffic = getattr(self.simulator, 'get_current_traffic_intensity', lambda x, y: 0.5)(creator.x,
                                                                                                     creator.y)

        # دریافت همزمان میانگین و بیشینه ترافیک مسیر آینده ماشین
        pred_avg, pred_max = getattr(self.simulator, 'get_predicted_traffic_intensity', lambda v: (0.5, 0.5))(creator)

        # آپدیت آب و هوا در استک (دریافت وضعیت فعلی آب و هوا: 0 خوب، 1 بد)
        current_weather = getattr(self.simulator, 'current_weather_status', 0.0)
        self.weather_history.append(current_weather)

        # ضریب تضعیف سیگنال (Path Loss) برای لوکیشن ماشین
        path_loss = getattr(self.simulator, 'get_path_loss', lambda x, y: 0.0)(creator.x, creator.y)

        # اضافه کردن هر سه پارامتر ترافیکی به استیت
        state_vector.extend([current_traffic, pred_avg, pred_max])
        state_vector.extend(list(self.weather_history))  # ۳ ویژگی آب و هوا
        state_vector.extend([path_loss])

        # ==========================================
        # بلوک ۴: ویژگی‌های 3 فاگ نزدیک
        # ==========================================
        nearest_fogs = self._get_k_nearest_fogs(creator, k=3)

        for i in range(3):
            if i < len(nearest_fogs):
                fog = nearest_fogs[i]
                dist = np.sqrt((fog.x - creator.x) ** 2 + (fog.y - creator.y) ** 2) / MAX_DISTANCE
                dist = min(dist, 1.0)  # محدود کردن روی 1

                f_rem_cap = fog.remaining_power / MAX_CAPACITY
                f_best_q = getattr(fog, 'get_best_queue_length', lambda: 0.0)() / MAX_QUEUE_LEN
                f_avg_q = getattr(fog, 'get_avg_queue_length', lambda: 0.0)() / MAX_QUEUE_LEN
                f_idle_cores = getattr(fog, 'get_idle_capable_cores_count', lambda t: 0.0)(task) / MAX_CORES

                state_vector.extend([dist, f_rem_cap, f_best_q, f_avg_q, f_idle_cores])
            else:
                # اگر فاگی وجود نداشت (مثلا فقط ۲ فاگ در کل نقشه بود)، بدترین حالت را پاس میدهیم
                state_vector.extend([1.0, 0.0, 1.0, 1.0, 0.0])

        # ==========================================
        # بلوک ۵: ویژگی‌های Cloud
        # ==========================================
        cloud = self.simulator.cloud_node
        if cloud:
            # todo: should fix this section. current_cloud_bandwidth doesn't need to write
            bw = getattr(self.simulator, 'current_cloud_bandwidth', MAX_BANDWIDTH) / MAX_BANDWIDTH
            est_delay = getattr(self.simulator, 'estimated_cloud_delay', 0.0) / MAX_DELAY
            c_rem_cap = cloud.remaining_power / MAX_CLOUD_CAPACITY
            c_best_q = getattr(cloud, 'get_best_queue_length', lambda: 0.0)() / MAX_QUEUE_LEN
            c_avg_q = getattr(cloud, 'get_avg_queue_length', lambda: 0.0)() / MAX_QUEUE_LEN
            c_idle_cores = getattr(cloud, 'get_idle_cores_count', lambda: 0.0)() / MAX_CORES

            state_vector.extend([bw, est_delay, c_rem_cap, c_best_q, c_avg_q, c_idle_cores])
        else:
            state_vector.extend([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])

        # print(red_bg(state_vector))
        return np.array(state_vector, dtype=np.float32)

    def _compute_reward_complex(self, task, executor):
        """
        Calculate combined rewards based on queue status, environmental
        conditions, and deadline compliance.
        """
        reward = 0.0

        idle_cores = getattr(executor, 'get_idle_capable_cores_count', lambda t: 0)(task)
        if idle_cores > 0:
            reward += 0.5
        else:
            best_queue = getattr(executor, 'get_best_queue_length', lambda: 0.0)()
            reward -= (best_queue * 0.1)

        # todo: surly i should change the logic of this section
        if executor != task.creator:
            weather = getattr(self.simulator, 'current_weather_status', 0.0)
            # todo: should fix this
            path_loss = getattr(self.simulator, 'get_path_loss', lambda x, y: 0.0)(task.creator.x, task.creator.y)

            env_penalty = (weather * 0.5) + (path_loss * 0.5)
            reward -= env_penalty

        # todo: should change this section
        isDeadlineMiss, lateness = isDeadlineMissHappening(task, executor, self.simulator.fixed_fog_nodes)

        if not isDeadlineMiss:
            reward += lateness * 1.0
        else:
            base_penalty = -50.0
            total_penalty = base_penalty + lateness
            reward += total_penalty

        return reward