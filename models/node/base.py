from __future__ import annotations

import abc
import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List

import numpy as np

from config import Config
from models.base import ModelBaseABC
from utils.enums import Layer
from utils.distance import get_distance

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


def find_closest_fn(x, y, fn_nodes, taskPower):
    closest_fn = None
    min_distance = float('inf')
    # print(fn_nodes)

    for fn in fn_nodes.values():
        # print(fn)
        distance = calculate_distance(x, y, fn.x, fn.y)
        if distance < min_distance:
            min_distance = distance
            closest_fn = fn

    return closest_fn


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
        return eta * Config.SimulatorConfig.BANDWIDTH * np.log2(1 + task.SNR)
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
    # todo : check if running_tasks is necessary
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

    def __post_init__(self):
        """
        This function runs automatically when you create a node.
        """
        # This IS the line that "creates the heaps"
        self.cores = [[] for _ in range(self.num_cores)]

        self.running_tasks = [None] * self.num_cores
        self.core_loads = [0.0] * self.num_cores

    def can_offload_task(self, task) -> bool:
        """Checks whether the task can be offloaded in this node."""
        if getattr(task, "is_hard", False):
            return False

        # note: i think this section could help drl and make a maximisation for each node
        if len(self.tasks) >= self.max_tasks_queue_len:
            # print(blue_bg(f"max_tasks_queue_len"))
            return False

        task_power = task.power
        if self.id == task.creator.id and self.layer == Layer.USER:
            task_power *= Config.UserNodeConfig.LOCAL_OFFLOAD_POWER_OVERHEAD

        # todo: maybe i should remove power and remaining power
        if task_power > self.remaining_power:
            # print(blue_bg(f"remaining_power"))
            return False

        if get_distance(self.x, self.y, task.creator.x, task.creator.y) > self.radius:
            # print(blue_bg(f"distance"))
            return False
        return True

    def get_transmission_time(self, task, fixed_fog_nodes) -> float:
        if task.creator.id != task.executor.id:
            # print(green_bg(f"dataRate = {dataRate}"))
            # print(self.id)
            if self.layer == Layer.FOG:
                dataRate = findDataRate(task, task.executor, 0)

                return task.dataSize / dataRate
                # print(blue_bg(f"executor: {task.executor.id}::: delay: {task.dataSize / dataRate}, dataRate: {dataRate}"))
            elif self.layer == Layer.CLOUD:

                closest_fn = find_closest_fn(task.creator.x, task.creator.y, fixed_fog_nodes, task.power)
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
        # todo: should change power concept
        # todo: add multicore and Worst Fit Decreasing assigning
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

    def __post_init__(self):
        super().__post_init__()
        self.radius = Config.MobileFogNodeConfig.DEFAULT_RADIUS
        self.tasks = deque()
        self.finished_tasks = deque()
        self.local_hard_tasks = []
        self.last_tbs_deadline = 0.0

    def assign_local_hard_task(self, task, current_time: float) -> None:
        """Register a hard task on this vehicle's local processor only."""
        task.creator = self
        task.executor = self
        task.creator_id = self.id
        task.release_time = current_time
        task.remaining_time = task.exec_time
        task.start_time = current_time
        task.is_hard = True
        self.local_hard_tasks.append(task)

    def _hard_task_utilization(self) -> float:
        """Worst-case utilization of pre-generated hard periodic tasks for TBS."""
        total_util = 0.0
        divisor = Config.UserNodeConfig.HARD_TASK_EXEC_TIME_DIVISOR
        for spec in Config.UserNodeConfig.HARD_TASK_SPECS:
            worst_exec = (spec["size_max"] * spec["cycles_max"]) / (self.frequency * divisor)
            total_util += worst_exec / spec["period"]
        return total_util

    def assign_task(self, task, current_time: float, fixed_fog_nodes) -> None:
        """Hard tasks → local processor. Soft tasks → TBS-scheduled local queue."""
        if task.is_hard:
            self.assign_local_hard_task(task, current_time)
            return

        task.executor = self
        task.release_time = current_time
        task.total_exec_time = findExecTimeInEachKindOfNode(task, executor=self)
        task.remaining_time = task.total_exec_time
        delay = self.get_transmission_time(task, fixed_fog_nodes)
        task.start_time = current_time + delay

        # TBS deadline assignment: d_k = max(r_k, d_{k-1}) + C_k / U_s.
        slack_util = max(0.1, 1.0 - self._hard_task_utilization())
        self.last_tbs_deadline = max(task.start_time, self.last_tbs_deadline) + (
            task.total_exec_time / slack_util
        )
        task.deadline = self.last_tbs_deadline
        self.tasks.append(task)

    def execute_tasks(self, current_time: float, fixed_fog_nodes) -> list:
        """Run one timestep of EDF over hard + TBS-scheduled soft tasks."""
        timestep = float(self.num_cores)
        finished_tasks_this_step = []
        ready_jobs = []

        for task in self.local_hard_tasks:
            if task.remaining_time > 0 and task.start_time <= current_time:
                heapq.heappush(ready_jobs, (task.deadline, task))

        for task in self.tasks:
            if task.remaining_time > 0 and task.start_time <= current_time:
                heapq.heappush(ready_jobs, (task.deadline, task))

        time_remaining = timestep
        while time_remaining > 1e-9 and ready_jobs:
            _, running_task = heapq.heappop(ready_jobs)
            run_time = min(running_task.remaining_time, time_remaining)
            running_task.remaining_time -= run_time
            time_remaining -= run_time

            if running_task.remaining_time <= 1e-9:
                running_task.finish_time = current_time + (timestep - time_remaining)
                finished_tasks_this_step.append(running_task)
                if running_task.is_hard:
                    self.local_hard_tasks = [
                        job for job in self.local_hard_tasks if job is not running_task
                    ]
                else:
                    try:
                        self.tasks.remove(running_task)
                    except ValueError:
                        pass
            else:
                heapq.heappush(ready_jobs, (running_task.deadline, running_task))

        self.finished_tasks.extend(finished_tasks_this_step)
        return finished_tasks_this_step
