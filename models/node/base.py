from __future__ import annotations

import abc
import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Dict
import xml.etree.ElementTree as ET

import numpy as np

from config import Config
from models.base import ModelBaseABC
from utils.enums import Layer
from utils.distance import get_distance
from models.task import Task


def blue_bg(text):
    return f"\033[44m{text}\033[0m"


def green_bg(text):
    return f"\033[42m{text}\033[0m"

def findExecTimeInEachKindOfNode(task, executor=None):
    from models.node.user import UserNode
    from models.node.cloud import CloudNode
    from models.node.fog import FixedFogNode
    from models.node.fog import MobileFogNode

    taskExecutor = task.executor
    if executor:
        taskExecutor = executor
    if isinstance(taskExecutor, UserNode):
        return task.real_exec_time(executor=taskExecutor)
    elif isinstance(taskExecutor, CloudNode):
        # print("CloudNode()")
        return task.real_exec_time(executor=taskExecutor) / (
                Config.CloudConfig.CLOUD_NODE_FREQUENCY / Config.UserNodeConfig.USER_NODE_FREQUENCY)
    elif isinstance(taskExecutor, FixedFogNode):
        # print("FixedFogNode()")
        return task.real_exec_time(executor=taskExecutor) / (
                Config.FixedFogNodeConfig.Fixed_NODE_FREQUENCY / Config.UserNodeConfig.USER_NODE_FREQUENCY)
    elif isinstance(taskExecutor, MobileFogNode):
        # print("MobileFogNode()")
        return task.real_exec_time(executor=taskExecutor) / (
                Config.MobileFogNodeConfig.MOBILE_NODE_FREQUENCY / Config.UserNodeConfig.USER_NODE_FREQUENCY)
    else:
        print(f"errrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrorr: {taskExecutor}")
        return -1


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def find_closest_fn(x, y, fn_nodes):
    closest_fn = None
    min_distance = float('inf')
    # print(fn_nodes)

    for fn in fn_nodes.values():
        # print(fn)
        distance = calculate_distance(x, y, fn.x, fn.y)
        if distance < min_distance:
            min_distance = distance
            closest_fn = fn

    return closest_fn, min_distance


def findDataRate(task, executor, closest_fn) -> float:
    from models.node.cloud import CloudNode
    from models.node.fog import FixedFogNode
    from models.node.fog import MobileFogNode
    eta = 0.0
    if isinstance(executor, MobileFogNode) or isinstance(executor, FixedFogNode):
        # print(blue_bg(f"{executor.id}:{len(executor.tasks)}"))
        # eta = task.power / executor.power
        eta = 1 / (len(executor.tasks) + 1)
    elif isinstance(executor, CloudNode):
        # eta = task.power / closest_fn.power
        eta = 1 / (len(closest_fn.tasks) + 1)

    # print(green_bg(
    #     f"task.id: {task.id}, task.executor.id: {executor.id}, task.power: {task.power}, task.executor.power: {executor.power}, eta : {eta}"))
    if task.SNR != 0:
        dataRate = eta * Config.SimulatorConfig.BANDWIDTH * np.log2(1 + task.SNR)
        # print(green_bg(f"executor: {task.executor.id} => eta: {eta}"))
        # print(blue_bg(f"DataRate: {dataRate}, DataSize:{task.dataSize} => {task.dataSize/dataRate}"))
        return dataRate
    else:
        return 1e-9


@dataclass
class NodeABC(ModelBaseABC, abc.ABC):
    """
    Represents any computational resource in the system. Nodes can belong to different layers,
    such as User, Fog, or Cloud.
    """

    x: float = 0
    y: float = 0
    power: float = 0  # The computational power available at this node.

    radius: float = 0  # The radius that this node can cover.
    frequency: float = 0
    # running_tasks: List[]

    remaining_power: float = 0  # The amount of computational resourced left after executing current tasks.
    tasks: Deque = field(default_factory=deque)  # The list of tasks that are currently offloaded in this node.
    finished_tasks: Deque = field(default_factory=deque)  # The list of tasks that are finished executing in this node.

    # Priority queues for WAITING tasks (one per core)
    # Used by assign_task and execute_tasks
    cores: List[List[tuple]] = field(init=False)
    # The total remaining work (load) for each core (for Worst Fit)
    # Used by assign_task and execute_tasks
    core_loads: List[float] = field(init=False)
    execution_log = None

    def __post_init__(self):
        """
        This function runs automatically when you create a node.
        """
        # This IS the line that "creates the heaps"
        self.cores = [[] for _ in range(self.num_cores)]

        # self.running_tasks = [None] * self.num_cores
        self.core_loads = [0.0] * self.num_cores

    def can_offload_task(self, task) -> bool:
        """Checks whether the task can be offloaded in this node."""
        # note: i think this section could help drl and make a maximisation for each node
        # todo: change queue limit number
        if len(self.tasks) >= self.max_tasks_queue_len:
            # print(blue_bg(f"max_tasks_queue_len"))
            # print(f"2222222222222222222222:{self.max_tasks_queue_len}")
            return False

        # note: i have removed power and remaining power constraint
        # if task_power > self.remaining_power:
        #     # print(blue_bg(f"remaining_power"))
        #     return False

        if get_distance(self.x, self.y, task.creator.x, task.creator.y) > self.radius:
            return False
        return True

    def get_best_queue_length(self) -> float:
        """
        Returns: The total remaining execution time (in seconds)
        of the least loaded core (the shortest queue by time, not by task count).
        """
        core_loads = [sum(task[-1].remaining_time for task in core) for core in self.cores]
        return float(min(core_loads))

    def get_avg_queue_length(self) -> float:
        """
        Returns: The average processing load (in seconds) across all cores at this node.
        """
        total_time_load = 0.0
        for i, core in enumerate(self.cores):
            core_load = sum(item[-1].remaining_time for item in core)

            # node_id = self.id
            # print(f"Node: {node_id} | Core {i} Load: {core_load:.4f} seconds")

            total_time_load += core_load
        return float(total_time_load) / max(1, self.num_cores)

    def get_idle_cores_count(self) -> float:
        """
        Returns: The exact number of cores that are currently IDLE (0 tasks currently running or waiting).
        """
        return float(sum(1 for core in self.cores if len(core) == 0))

    def get_idle_capable_cores_count(self, task) -> float:
        """
        Returns: The number of idle cores that possess the required processing power for this specific task.
        """
        if not self.can_offload_task(task):
            # print(blue_bg("*******************"))
            return 0.0

        return self.get_idle_cores_count()

    def get_transmission_time(self, task, fixed_fog_nodes) -> float:
        if task.creator.id != task.executor.id:
            # print(green_bg(f"dataRate = {dataRate}"))
            # print(self.id)
            if self.layer == Layer.FOG:
                dataRate = findDataRate(task, task.executor, 0)

                return task.dataSize / dataRate
                # print(blue_bg(f"executor: {task.executor.id}::: delay: {task.dataSize / dataRate}, dataRate: {dataRate}"))
            elif self.layer == Layer.CLOUD:

                closest_fn, _ = find_closest_fn(task.creator.x, task.creator.y, fixed_fog_nodes)
                dataRate = findDataRate(task, task.executor, closest_fn)

                if closest_fn.x == Config.CloudConfig.CLOSEST_FOG_X and closest_fn.y == Config.CloudConfig.CLOSEST_FOG_Y:
                    return (task.dataSize / dataRate) + (
                            task.dataSize / Config.CloudConfig.CLOUD_BANDWIDTH)
                    # print(blue_bg(f"executor: {task.executor.id}::: delay: {(task.dataSize / dataRate) + (task.dataSize / Config.CloudConfig.CLOUD_BANDWIDTH)}, dataRate: {dataRate}"))
                else:
                    return (task.dataSize / dataRate) + 2 * (
                            task.dataSize / Config.CloudConfig.CLOUD_BANDWIDTH)
                    # print(blue_bg(f"executor: {task.executor.id}:::{task.id} firstStepDelay: {(task.dataSize / dataRate)}, delay: {(task.dataSize / dataRate) + 2 * (task.dataSize / Config.CloudConfig.CLOUD_BANDWIDTH)}, dataRate: {dataRate}"))
        else:
            return 0.0

    def assign_task(self, task, current_time: float, fixed_fog_nodes) -> None:
        """Offload a task in the current node."""
        # 1. Initialize task execution parameters
        # Calculate the TOTAL discrete time steps this task needs to complete
        self.tasks.append(task)
        task.executor = self
        task.total_exec_time = findExecTimeInEachKindOfNode(task)
        task.remaining_time = task.total_exec_time

        delay = self.get_transmission_time(task, fixed_fog_nodes)
        task.start_time = current_time + delay
        # 2. Worst Fit Selection: Find the core with the minimum current load
        least_loaded_core_idx = self.core_loads.index(min(self.core_loads))
        self.core_loads[least_loaded_core_idx] += task.total_exec_time
        heapq.heappush(self.cores[least_loaded_core_idx], (task.deadline, task.release_time, task))

    def execute_tasks(self, current_time: float, fixed_fog_nodes) -> list:
        """
            Executes tasks on all cores for one time step using Preemptive EDF.
            Relies on a heap to manage task priority by deadline.
            - Always runs the task with the earliest deadline.
            - Skips tasks whose start_time > current_time.
            - If a task finishes before the tick ends, continues with the next ready task.
        """
        if self.execution_log is None:
            self.execution_log = []
        finished_tasks_this_step = []
        WORK_PER_TICK = 1.0

        for i in range(self.num_cores):
            remaining_work_this_tick = WORK_PER_TICK
            core_heap = self.cores[i]
            temp_unready_tasks = []

            while remaining_work_this_tick > 0:
                if not core_heap:
                    break

                deadline, rel_time, task = heapq.heappop(core_heap)

                if task.start_time > current_time:
                    temp_unready_tasks.append((deadline, rel_time, task))
                    continue

                work_to_do = min(remaining_work_this_tick, task.remaining_time)

                slice_start = current_time + (WORK_PER_TICK - remaining_work_this_tick)
                self.execution_log.append({
                    'core': i,
                    'task_id': task.id,
                    'start': slice_start,
                    'duration': work_to_do,
                    'is_hard': getattr(task, 'is_hard', False)
                })

                task.remaining_time -= work_to_do
                self.core_loads[i] -= work_to_do
                remaining_work_this_tick -= work_to_do

                if task.remaining_time <= 0:
                    task.finish_time = current_time + (WORK_PER_TICK - remaining_work_this_tick)
                    finished_tasks_this_step.append(task)
                    self.finished_tasks.append(task)

                    if task in self.tasks:
                        self.tasks.remove(task)

                    continue
                else:
                    heapq.heappush(core_heap, (deadline, rel_time, task))
                    break

            for item in temp_unready_tasks:
                heapq.heappush(core_heap, item)

        return finished_tasks_this_step

    @property
    @abc.abstractmethod
    def layer(self) -> Layer:
        """The layer to which this node belongs (User, Fog, or Cloud)."""
        raise NotImplemented

    @property
    @abc.abstractmethod
    def max_tasks_queue_len(self) -> int:
        """The max number of tasks that this node can process in parallel."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_cores(self) -> int:
        """The number of processing cores available in this node."""
        raise NotImplementedError

@dataclass
class MobileNodeABC(NodeABC, abc.ABC):
    """
    Represents any computational resource in the system which can move from one location to another. Mobile nodes may
    belong to different layers, such as User or Fog.
    """

    speed: float = 0
    angle: float = 0
    weather: str = ""

    core_Up: List[float] = field(init=False)
    core_Us: List[float] = field(init=False)
    last_tbs_deadline: List[float] = field(init=False)
    periodic_allocation: List[List[dict]] = field(init=False)
    local_hard_tasks = []

    def __post_init__(self):
        super().__post_init__()
        self.core_Up = [0.0] * self.num_cores
        self.core_Us = [1.0] * self.num_cores
        self.last_tbs_deadline = [0.0] * self.num_cores
        self.periodic_allocation = [[] for _ in range(self.num_cores)]

    def assign_local_hard_task(self, task, current_time: float) -> None:
        """Register a hard task on this vehicle's local processor using Worst Fit (Utilization-based)."""
        task.creator = self
        task.executor = self
        task.creator_id = self.id
        task.release_time = current_time

        exec_time_calculated = findExecTimeInEachKindOfNode(task)
        task.total_exec_time = exec_time_calculated if exec_time_calculated > 0 else task.exec_time
        task.remaining_time = task.total_exec_time
        task.start_time = current_time
        task.is_hard = True
        self.local_hard_tasks.append(task)


        period = task.deadline - task.release_time
        task_utilization = task.total_exec_time / period if period > 0 else task.total_exec_time

        best_core_idx = self.core_Up.index(min(self.core_Up))

        self.core_Up[best_core_idx] += task_utilization

        self.core_Us[best_core_idx] = max(0.01, 1.0 - self.core_Up[best_core_idx])

        self.core_loads[best_core_idx] += task.total_exec_time

        heapq.heappush(self.cores[best_core_idx], (task.deadline, task.release_time, task))

    def assign_task(self, task, current_time: float, fixed_fog_nodes=None) -> None:
        self.tasks.append(task)
        task.executor = self
        task.total_exec_time = findExecTimeInEachKindOfNode(task)
        task.remaining_time = task.total_exec_time

        delay = self.get_transmission_time(task, fixed_fog_nodes)

        task.start_time = current_time + delay
        rk = task.start_time

        best_core_idx = 0
        min_prospective_deadline = float('inf')

        for i in range(self.num_cores):
            Us = self.core_Us[i]
            last_dl = self.last_tbs_deadline[i]
            Ck = task.total_exec_time

            prospective_dk = max(rk, last_dl) + (Ck / Us)

            if prospective_dk < min_prospective_deadline:
                min_prospective_deadline = prospective_dk
                best_core_idx = i

        self.last_tbs_deadline[best_core_idx] = min_prospective_deadline

        self.core_loads[best_core_idx] += task.total_exec_time

        heapq.heappush(self.cores[best_core_idx], (min_prospective_deadline, task.release_time, task))

    def execute_tasks(self, current_time: float, fixed_fog_nodes) -> list:
        """
            Executes tasks on all cores for one time step using Preemptive EDF.
            Relies on a heap to manage task priority by deadline.
            - Always runs the task with the earliest deadline.
            - Skips tasks whose start_time > current_time.
            - If a task finishes before the tick ends, continues with the next ready task.
        """
        if self.execution_log is None:
            self.execution_log = []
        finished_tasks_this_step = []
        WORK_PER_TICK = 1.0

        for i in range(self.num_cores):
            remaining_work_this_tick = WORK_PER_TICK
            core_heap = self.cores[i]
            temp_unready_tasks = []

            while remaining_work_this_tick > 0:
                if not core_heap:
                    break

                deadline, rel_time, task = heapq.heappop(core_heap)

                if task.start_time > current_time:
                    temp_unready_tasks.append((deadline, rel_time, task))
                    continue

                work_to_do = min(remaining_work_this_tick, task.remaining_time)

                slice_start = current_time + (WORK_PER_TICK - remaining_work_this_tick)
                self.execution_log.append({
                    'core': i,
                    'task_id': task.id,
                    'start': slice_start,
                    'duration': work_to_do,
                    'is_hard': getattr(task, 'is_hard', False)
                })

                task.remaining_time -= work_to_do
                self.core_loads[i] -= work_to_do
                remaining_work_this_tick -= work_to_do

                if task.remaining_time <= 0:
                    task.finish_time = current_time + (WORK_PER_TICK - remaining_work_this_tick)
                    finished_tasks_this_step.append(task)
                    self.finished_tasks.append(task)

                    if task in self.tasks:
                        self.tasks.remove(task)

                    if task in self.local_hard_tasks:
                        self.local_hard_tasks.remove(task)

                    continue
                else:
                    heapq.heappush(core_heap, (deadline, rel_time, task))
                    break

            for item in temp_unready_tasks:
                heapq.heappush(core_heap, item)

        return finished_tasks_this_step
