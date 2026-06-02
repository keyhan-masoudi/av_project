import csv
import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from config import Config
from models.node.cloud import CloudNode
from task_and_user_generator import Config as CNF
from models.node.base import findExecTimeInEachKindOfNode, find_closest_fn, findDataRate, green_bg
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
        # 28: Task(3) + Local(3) + Env(7) + Fog(12) + Cloud(3)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(28,), dtype=np.float32
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

    def get_action_mask(self, task):
        # [Local, Cloud, Fog1, Fog2, Fog3]
        mask = np.ones(5, dtype=np.float32)

        creator = task.creator

        # =====================================
        # Local Masking
        # =====================================
        local_best_q = creator.get_best_queue_length()
        if (local_best_q / CNF.TaskConfig.DEADLINE_MAX_FREE_TIME) > 0.85:
            mask[0] = 0.0

        # todo: maybe it won't be bad if we mask fog with queue too
        # =====================================
        # Masking Fogs considering their coverage
        # =====================================
        nearest_fogs = self._get_k_nearest_fogs(creator, k=3)

        for i in range(3):
            action_idx = i + 2

            if i < len(nearest_fogs):
                fog = nearest_fogs[i]
                dist = np.sqrt((fog.x - creator.x) ** 2 + (fog.y - creator.y) ** 2)

                if dist > creator.radius:
                    mask[action_idx] = 0.0
            else:
                mask[action_idx] = 0.0

        return mask

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

        if candidate_executor.can_offload_task(task):
            print(green_bg("are we here?????????????????????????????????"))
            reward = self._compute_reward(task, candidate_executor)
        else:
            reward = -10.0
        return reward

    def _get_k_nearest_fogs(self, vehicle, k=3):
        all_fogs = list(self.simulator.fixed_fog_nodes.values()) + list(self.simulator.mobile_fog_nodes.values())
        all_fogs.sort(key=lambda fog: np.sqrt((fog.x - vehicle.x) ** 2 + (fog.y - vehicle.y) ** 2))
        return all_fogs[:k]

    def _get_state(self, task=None, current_time=None):
        if task is None:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        state_vector = []
        creator = task.creator

        # ==========================================
        # Task
        # ==========================================
        data_size_ratio = task.dataSize / CNF.TaskConfig.MAX_DATASIZE
        workload_ratio = task.power / CNF.TaskConfig.MAX_POWER_CONSUMPTION
        deadline_ratio = min(((task.deadline - current_time) / CNF.TaskConfig.DEADLINE_MAX_FREE_TIME), 1)

        state_vector.extend([data_size_ratio, workload_ratio, deadline_ratio])

        # ==========================================
        # Local Node
        # ==========================================
        # local_tot_cap = creator.power / Config.FixedFogNodeConfig.DEFAULT_COMPUTATION_POWER
        # local_rem_cap = creator.remaining_power / Config.FixedFogNodeConfig.DEFAULT_COMPUTATION_POWER

        # if creator.id == "PKW135":
        #     print(green_bg(
        #         f"Local:\nget_best_queue_length: {creator.get_best_queue_length()}, get_avg_queue_length: {creator.get_avg_queue_length()}, get_idle_cores_count: {creator.get_idle_cores_count()}"))
        local_best_q = min(creator.get_best_queue_length() / CNF.TaskConfig.DEADLINE_MAX_FREE_TIME, 1.0)
        local_avg_q = min(creator.get_avg_queue_length() / CNF.TaskConfig.DEADLINE_MAX_FREE_TIME, 1.0)
        local_idle_cores = -1

        local_idle_cores = creator.get_idle_cores_count() / len(creator.cores)

        # state_vector.extend([local_tot_cap, local_rem_cap, local_best_q, local_avg_q, local_idle_cores])
        state_vector.extend([local_best_q, local_avg_q, local_idle_cores])

        # ==========================================
        # (Environment & Context)
        # ==========================================
        current_traffic = self.simulator.get_current_traffic_intensity(creator.x, creator.y)

        pred_avg, pred_max = self.simulator.get_predicted_traffic_intensity(creator)

        current_weather = self.simulator.current_weather_status
        self.weather_history.append(current_weather)

        # Extract the environmental path loss exponent (n) for the vehicle's current location
        # check: is it okay?
        n_coefficient = self.simulator.get_n_coefficient(creator.x, creator.y)

        state_vector.extend([current_traffic, pred_avg, pred_max])
        state_vector.extend(list(self.weather_history))
        state_vector.extend([n_coefficient])

        # ==========================================
        # Fogs Futures
        # ==========================================
        nearest_fogs = self._get_k_nearest_fogs(creator, k=3)

        for i in range(3):
            if i < len(nearest_fogs):
                fog = nearest_fogs[i]
                dist = np.sqrt(
                    (fog.x - creator.x) ** 2 + (fog.y - creator.y) ** 2) / Config.MobileFogNodeConfig.DEFAULT_RADIUS
                dist = min(dist, 1.0)

                # f_rem_cap = fog.remaining_power / fog.power

                #     if creator.id == "PKW135":
                #         print(green_bg(
                #             f"Fog{i}:\nget_best_queue_length: {fog.get_best_queue_length()}, get_avg_queue_length: {fog.get_avg_queue_length()}, get_idle_cores_count: {fog.get_idle_cores_count()}"))
                f_best_q = fog.get_best_queue_length() / CNF.TaskConfig.DEADLINE_MAX_FREE_TIME
                f_avg_q = fog.get_avg_queue_length() / CNF.TaskConfig.DEADLINE_MAX_FREE_TIME
                f_idle_cores = fog.get_idle_capable_cores_count(task) / len(fog.cores)

                state_vector.extend([dist, f_best_q, f_avg_q, f_idle_cores])
            else:
                # pass the worst state if there is not enough fog
                state_vector.extend([1.0, 1.0, 1.0, 0.0])

        # ==========================================
        # Cloud
        # ==========================================
        cloud = self.simulator.cloud_node
        if cloud:
            # c_rem_cap = cloud.remaining_power / cloud.power
            # if creator.id == "PKW135":
            #     print(green_bg(
            #         f"Cloud:\nget_best_queue_length: {cloud.get_best_queue_length()}, get_avg_queue_length: {cloud.get_avg_queue_length()}, get_idle_cores_count: {cloud.get_idle_cores_count()}"))
            c_best_q = cloud.get_best_queue_length() / CNF.TaskConfig.DEADLINE_MAX_FREE_TIME
            c_avg_q = cloud.get_avg_queue_length() / CNF.TaskConfig.DEADLINE_MAX_FREE_TIME
            c_idle_cores = cloud.get_idle_capable_cores_count(task) / len(cloud.cores)

            state_vector.extend([c_best_q, c_avg_q, c_idle_cores])
        else:
            state_vector.extend([1.0, 0.0, 1.0, 1.0, 0.0])

        # print(red_bg(state_vector))
        return np.array(state_vector, dtype=np.float32)

    def get_action_from_executor(self, task, executor) -> int:
        """
        Maps the selected executor back to the discrete action space (0 to 4).
        Action 0: Local execution.
        Action 1: Cloud execution.
        Action 2, 3, 4: Nearest fog nodes.
        """
        from models.node.cloud import CloudNode

        if executor.id == task.creator.id:
            return 0
        elif isinstance(executor, CloudNode):
            return 1
        else:
            # Find which of the 3 nearest fogs was selected
            nearest_fogs = self._get_k_nearest_fogs(task.creator, k=3)
            for i, fog in enumerate(nearest_fogs):
                if fog.id == executor.id:
                    return i + 2

            # Fallback (Should not occur if the mask logic is correct)
            return 0

    # todo: complete reward function
    def _compute_reward(self, task, executor) -> float:
        """
        Calculates the REAL reward using a combination of Reward Shaping (from saved state)
        and Ground Truth (from actual completion time).
        """
        reward = 0.0

        # ==========================================================
        # 1. Reward Shaping: Based on the state AT THE TIME OF DECISION
        # ==========================================================
        if hasattr(task, 'rl_state') and hasattr(task, 'rl_action'):
            state = task.rl_state
            action = task.rl_action

            # Map the action to the exact indices in the 28-dimensional state array
            # Format -> Action: (best_q_index, idle_cores_index)
            state_indices = {
                0: (3, 5),  # Local
                1: (25, 27),  # Cloud
                2: (14, 16),  # Fog 1
                3: (18, 20),  # Fog 2
                4: (22, 24)  # Fog 3
            }

            if action in state_indices:
                q_idx, idle_idx = state_indices[action]
                normalized_best_q = state[q_idx]
                normalized_idle_cores = state[idle_idx]

                if normalized_idle_cores > 0:
                    # Reward for choosing a node with completely free cores
                    # print(red_bg(
                    #     f"---------------best_queue: {normalized_best_q}, executor: {executor.id}"))
                    reward += 0.5
                else:
                    # Penalty based on how full the queue was (normalized 0.0 to 1.0)
                    # We multiply by 5.0 to give it a meaningful weight in the reward function
                    # print(red_bg(
                    #     f"+++++++++++++++++++++++++++++++++++++++best_queue: {normalized_best_q}, executor: {executor.id}"))
                    reward -= (normalized_best_q * 5.0)

        # ==========================================================
        # 2. Ground Truth Reward: Based on ACTUAL execution results
        # ==========================================================
        is_deadline_miss = task.finish_time > task.deadline
        lateness = task.deadline - task.finish_time  # Positive means early, negative means late

        if not is_deadline_miss:
            # Positive reward for finishing early
            reward += lateness * 1.0
        else:
            # Heavy penalty for missing the deadline, plus the amount of lateness
            base_penalty = -50.0
            reward += (base_penalty + lateness)

        # ==========================================================
        # 3. Environmental Penalty (Network/Transmission cost)
        # ==========================================================
        if executor.id != task.creator.id:
            weather = self.simulator.current_weather_status

            # Get the path loss exponent dynamically based on the urban area
            n_coefficient = getattr(self.simulator, 'get_n_coefficient', lambda x, y: 2.0)(task.creator.x,
                                                                                           task.creator.y)

            env_penalty = (weather * 0.5) + (n_coefficient * 0.5)
            reward -= env_penalty

        return reward