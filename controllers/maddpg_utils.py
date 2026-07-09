import numpy as np
import math
from collections import deque
from config import Config
from models.node.base import findExecTimeInEachKindOfNode, findDataRate, find_closest_fn
from models.node.fog import FixedFogNode, MobileFogNode
from task_and_user_generator import Config as Cnf
from NoiseConfigs.noiseConfigGeneralAttribute import NoiseConfigGeneralAttribute as NCNF

def get_action_mask(task, simulator):
    """
    Returns a mask vector [Local, Cloud, Fog1, Fog2, Fog3] (Dim=5).
    """
    mask = np.ones(5, dtype=np.float32)
    creator = task.creator

    # =====================================
    # Local Masking
    # =====================================
    local_best_q = creator.get_best_queue_length()
    if (local_best_q / Cnf.TaskConfig.DEADLINE_MAX_FREE_TIME) > 0.85:
        mask[0] = 0.0

    # =====================================
    # Masking Fogs considering their coverage
    # =====================================
    all_fogs = list(simulator.fixed_fog_nodes.values()) + list(simulator.mobile_fog_nodes.values())
    all_fogs.sort(key=lambda fog: np.sqrt((fog.x - creator.x) ** 2 + (fog.y - creator.y) ** 2))
    nearest_fogs = all_fogs[:3]

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


def get_agent_state(task, simulator):
    """
    Returns the exact 28-dimensional state vector identical to DeepRLEnvironment._get_state().
    """
    if task is None:
        return np.zeros(28, dtype=np.float32)

    state_vector = []
    creator = task.creator
    current_time = simulator.clock.get_current_time()

    # ==========================================
    # 1. Task (3 Dims)
    # ==========================================
    data_size_ratio = max(0.0, min((task.dataSize - Cnf.TaskConfig.MIN_DATASIZE) / (
            Cnf.TaskConfig.MAX_DATASIZE - Cnf.TaskConfig.MIN_DATASIZE), 1.0))

    min_exec = (
                           Cnf.TaskConfig.MIN_DATASIZE * Cnf.TaskConfig.MIN_CYCLE_PER_BIT) / Config.UserNodeConfig.USER_NODE_FREQUENCY
    max_exec = (
                           Cnf.TaskConfig.MAX_DATASIZE * Cnf.TaskConfig.MAX_CYCLE_PER_BIT) / Config.UserNodeConfig.USER_NODE_FREQUENCY
    execution_time_ratio = max(0.0, min((task.exec_time - min_exec) / (max_exec - min_exec), 1.0))

    deadline_ratio = max(0.0, min((((task.deadline - current_time) - Cnf.TaskConfig.DEADLINE_MIN_FREE_TIME) / (
            Cnf.TaskConfig.DEADLINE_MAX_FREE_TIME - Cnf.TaskConfig.DEADLINE_MIN_FREE_TIME)), 1.0))

    state_vector.extend([data_size_ratio, execution_time_ratio, deadline_ratio])

    # ==========================================
    # 2. Local Node (3 Dims)
    # ==========================================
    local_best_q = min(creator.get_best_queue_length() / Cnf.TaskConfig.DEADLINE_MAX_FREE_TIME, 1.0)
    local_avg_q = min(creator.get_avg_queue_length() / Cnf.TaskConfig.DEADLINE_MAX_FREE_TIME, 1.0)
    local_idle_cores = creator.get_idle_cores_count() / len(creator.cores)
    state_vector.extend([local_best_q, local_avg_q, local_idle_cores])

    # ==========================================
    # 3. Environment & Context (7 Dims)
    # ==========================================
    current_traffic = simulator.get_current_traffic_intensity(creator.x, creator.y)
    pred_avg, pred_max = simulator.get_predicted_traffic_intensity(creator)

    current_weather = simulator.get_current_weather(creator.x, creator.y) / (len(NCNF.Rain_options) - 1)

    # Handle weather history caching directly in simulator
    if not hasattr(simulator, 'weather_history'):
        simulator.weather_history = deque([0.0, 0.0, 0.0], maxlen=3)
    simulator.weather_history.append(current_weather)

    n_coefficient = simulator.get_n_coefficient(creator.x, creator.y)
    max_n = Config.AttenuationLevel.DEFAULT_AttenuationLevel[-1]
    min_n = Config.AttenuationLevel.DEFAULT_AttenuationLevel[0]
    normalized_n = (n_coefficient - min_n) / (max_n - min_n)

    state_vector.extend([current_traffic, pred_avg, pred_max])
    state_vector.extend(list(simulator.weather_history))
    state_vector.extend([normalized_n])

    # ==========================================
    # 4. Fogs Futures (12 Dims)
    # ==========================================
    all_fogs = list(simulator.fixed_fog_nodes.values()) + list(simulator.mobile_fog_nodes.values())
    all_fogs.sort(key=lambda fog: np.sqrt((fog.x - creator.x) ** 2 + (fog.y - creator.y) ** 2))
    nearest_fogs = all_fogs[:3]

    for i in range(3):
        if i < len(nearest_fogs):
            fog = nearest_fogs[i]
            dist = math.sqrt((creator.x - fog.x) ** 2 + (creator.y - fog.y) ** 2)
            normalized_dist = min((dist / Config.MobileFogNodeConfig.DEFAULT_RADIUS), 1)

            f_best_q = min((fog.get_best_queue_length() / Cnf.TaskConfig.DEADLINE_MAX_FREE_TIME), 1)
            f_avg_q = min((fog.get_avg_queue_length() / Cnf.TaskConfig.DEADLINE_MAX_FREE_TIME), 1)
            f_idle_cores = fog.get_idle_capable_cores_count(task) / len(fog.cores)

            state_vector.extend([normalized_dist, f_best_q, f_avg_q, f_idle_cores])
        else:
            state_vector.extend([1.0, 1.0, 1.0, 0.0])

    # ==========================================
    # 5. Cloud (3 Dims)
    # ==========================================
    cloud = simulator.cloud_node
    if cloud:
        c_best_q = min(cloud.get_best_queue_length() / Cnf.TaskConfig.DEADLINE_MAX_FREE_TIME, 1.0)
        c_avg_q = min(cloud.get_avg_queue_length() / Cnf.TaskConfig.DEADLINE_MAX_FREE_TIME, 1.0)
        c_idle_cores = cloud.get_idle_capable_cores_count(task) / len(cloud.cores)
        state_vector.extend([c_best_q, c_avg_q, c_idle_cores])
    else:
        state_vector.extend([1.0, 1.0, 0.0])

    return np.array(state_vector, dtype=np.float32)


def compute_agent_reward(task, executor, all_fog_nodes):
    """
    Calculates the REAL delayed reward combining shaping and ground truth.
    Extracts MADDPG specific stored states/actions.
    """
    reward = 0.0
    lateness = 0.0

    if hasattr(task, 'maddpg_ordered_states') and hasattr(task, 'maddpg_actions') and hasattr(task,
                                                                                              'maddpg_chosen_agent_id'):
        # Get the state and action specifically chosen by the winning agent
        state = task.maddpg_ordered_states[task.maddpg_chosen_agent_id]
        action = task.maddpg_actions[task.maddpg_chosen_agent_id]

        # ==========================================================
        # 1. Reward Shaping
        # ==========================================================
        state_indices = {
            0: (3, 5),  # Local Execution
            1: (25, 27),  # Cloud Execution
            2: (14, 16),  # Fog 1 Execution
            3: (18, 20),  # Fog 2 Execution
            4: (22, 24)  # Fog 3 Execution
        }

        if action in state_indices:
            q_idx, idle_idx = state_indices[action]
            normalized_best_q = state[q_idx]
            normalized_idle_cores = state[idle_idx]

            if normalized_idle_cores > 0:
                reward += 1.5
            else:
                reward -= (normalized_best_q * 5.0)

        # ==========================================================
        # 2. Dynamic Contextual Penalties (Local vs. Network)
        # ==========================================================
        current_traffic = state[6]
        traffic_avg = state[7]
        traffic_max = state[8]
        normalized_weather = state[11]
        normalized_n = state[12]

        traffic_component = (traffic_avg * 0.75) + (traffic_max * 0.25)
        local_stress = (traffic_component * 0.5) + (normalized_weather * 0.5)
        network_stress = (normalized_n * 0.75) + (normalized_weather * 0.25)

        if action == 0:
            if local_stress > 0.75:
                reward -= (local_stress ** 2) * 5.0
        else:
            if network_stress > 0.5:
                reward -= (network_stress ** 2) * 5.0
            if local_stress > 0.5 and local_stress > network_stress:
                courage_bonus = local_stress - network_stress
                reward += (courage_bonus * 5.0)

        # ==========================================================
        # 3. Ground Truth Reward (Based on actual execution)
        # ==========================================================
        is_deadline_miss = task.finish_time > task.deadline
        lateness = task.deadline - task.finish_time

        if not is_deadline_miss:
            reward += lateness
        else:
            base_penalty = -15.0
            reward += (base_penalty + lateness)

            # Environmental Penalty (Network/Transmission cost)
            if executor.id != task.creator.id:
                if action == 2:
                    normalized_dist = state[13]
                elif action == 3:
                    normalized_dist = state[17]
                elif action == 4:
                    normalized_dist = state[21]
                elif action == 1:
                    _, normalized_dist = find_closest_fn(task.creator.x, task.creator.y, all_fog_nodes)
                else:
                    dist = math.sqrt((task.creator.x - executor.x) ** 2 + (task.creator.y - executor.y) ** 2)
                    normalized_dist = min((dist / Config.MobileFogNodeConfig.DEFAULT_RADIUS), 1.0)

                base_env_penalty = (normalized_n * 0.5) + (normalized_weather * 0.25) + (current_traffic * 0.25)
                distance_factor = 0.2 + (0.8 * normalized_dist)
                env_penalty = base_env_penalty * distance_factor

                alpha = 3.0
                reward -= alpha * env_penalty

    return reward