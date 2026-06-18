import numpy as np
from config import Config
from models.node.base import findExecTimeInEachKindOfNode, findDataRate, find_closest_fn
from models.node.fog import FixedFogNode, MobileFogNode
from task_and_user_generator import Config as Cnf


def is_deadline_miss_happening(task, executor, fn_nodes):
    task.real_exec_time_base = findExecTimeInEachKindOfNode(task, executor)
    real_exec_time = task.real_exec_time_base

    if executor == task.creator:
        lateness = task.deadline - (task.release_time + real_exec_time)
        return lateness < 0, lateness

    elif isinstance(executor, (FixedFogNode, MobileFogNode)):
        data_rate = findDataRate(task, executor, 0)
        real_exec_time += task.dataSize / data_rate
        lateness = task.deadline - (task.release_time + real_exec_time)
        return lateness < 0, lateness

    else:  # CloudNode
        closest_fn, _ = find_closest_fn(task.creator.x, task.creator.y, fn_nodes)
        dataRate = findDataRate(task, executor, closest_fn)
        # print(f"closest_fn:{closest_fn}, x: {closest_fn}")
        if closest_fn.x == Config.CloudConfig.CLOSEST_FOG_X and closest_fn.y == Config.CloudConfig.CLOSEST_FOG_Y:
            real_exec_time += (task.dataSize / dataRate) + (
                    task.dataSize / Config.CloudConfig.CLOUD_BANDWIDTH)
        else:
            real_exec_time += (task.dataSize / dataRate) + 2 * (
                    task.dataSize / Config.CloudConfig.CLOUD_BANDWIDTH)

        return ((task.release_time + real_exec_time) > task.deadline), (
                task.deadline - (task.release_time + real_exec_time))

def compute_agent_reward(task, executor, fn_nodes):
    if executor is None or not executor.can_offload_task(task):
        return Config.NEGATIVE_REWARD

    is_missed, lateness = is_deadline_miss_happening(task, executor, fn_nodes)

    if is_missed:
        return -5.0 + (lateness / task.deadline)
    else:
        return 5.0 * (lateness / task.deadline)

def get_action_mask(task):
    """
    Returns a mask vector [local, fog, cloud].
    If local execution misses the deadline, mask[0] becomes 0.
    """
    mask = [1.0, 1.0, 1.0]
    if task is not None:
        local_exec_time = findExecTimeInEachKindOfNode(task, task.creator)

        if (task.release_time + local_exec_time) > task.deadline:
            mask[0] = 0.0

    return np.array(mask, dtype=np.float32)

def get_agent_state(task, simulator):
    creator = task.creator
    # maxSpeedOfaVehicle = 13.89

    vehicle_power_ratio = creator.remaining_power / creator.power
    task_power_ratio = task.power / Cnf.TaskConfig.MAX_POWER_CONSUMPTION
    exec_time_ratio = task.exec_time / Cnf.TaskConfig.MAX_EXEC_TIME
    # vehicle_speed_ratio = creator.speed / maxSpeedOfaVehicle

    max_fog_power = 19.79

    nearby_fogs = [
        node for node in simulator.fixed_fog_nodes.values()
        if np.sqrt((node.x - creator.x) ** 2 + (node.y - creator.y) ** 2) <= 300
    ]
    if not nearby_fogs:
        avg_fog_power_ratio = 0.0
    else:
        avg_power = sum(node.remaining_power for node in nearby_fogs) / len(nearby_fogs)
        avg_fog_power_ratio = avg_power / max_fog_power

    cloud_power_ratio = simulator.cloud_node.remaining_power / simulator.cloud_node.power

    return np.array([
        vehicle_power_ratio,
        task_power_ratio,
        exec_time_ratio,
        avg_fog_power_ratio,
        # vehicle_speed_ratio,
        cloud_power_ratio
    ], dtype=np.float32)