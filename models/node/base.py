from __future__ import annotations

import abc
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque
from typing import List

import xml.etree.ElementTree as ET
import heapq

import numpy as np
import random

from config import Config
from controllers.metric import MetricsController
from models.base import ModelBaseABC
# from models.node.user import CriticalUserNode
from utils.enums import Layer
from utils.distance import get_distance
from typing import TYPE_CHECKING
from models.task import Task 

PERIODIC_TASKS = [
    (7.0,  (800.0, 1200.0), (1000.0, 1200.0)),
    (5.0,  (1000.0, 5000.0), (1000.0, 1200.0)),
    (6.0,  (500.0, 1000.0),  (500.0, 1000.0)),
]
random.seed(42)

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
        # print("UserNode()")
        return task.real_exec_time(executor=taskExecutor)
    elif isinstance(taskExecutor, CriticalUserNode):
        return task.real_exec_time(executor=taskExecutor) / (Config.CriticalUserNodeConfig.USER_NODE_FREQUENCY / Config.UserNodeConfig.USER_NODE_FREQUENCY)
    elif isinstance(taskExecutor, CloudNode):
        # print("CloudNode()")
        return task.real_exec_time(executor=taskExecutor) / (Config.CloudConfig.CLOUD_NODE_FREQUENCY / Config.UserNodeConfig.USER_NODE_FREQUENCY)
    elif isinstance(taskExecutor, FixedFogNode):
        # print("FixedFogNode()")
        return task.real_exec_time(executor=taskExecutor) / (Config.FixedFogNodeConfig.Fixed_NODE_FREQUENCY / Config.UserNodeConfig.USER_NODE_FREQUENCY)
    elif isinstance(taskExecutor, MobileFogNode):
        # print("MobileFogNode()")
        return task.real_exec_time(executor=taskExecutor) / (Config.MobileFogNodeConfig.MOBILE_NODE_FREQUENCY / Config.UserNodeConfig.USER_NODE_FREQUENCY)
    else:
        print("errrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrorr")
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
        # print("----------------------------------test----------------------------------")
        # todo : improve this part
        if len(self.tasks) >= self.max_tasks_queue_len:
            # print(blue_bg(f"max_tasks_queue_len"))
            return False
        task_power = task.power
        if self.id == task.creator.id and self.layer == Layer.USER:
            task_power *= Config.UserNodeConfig.LOCAL_OFFLOAD_POWER_OVERHEAD
        if task_power > self.remaining_power:
            # print(blue_bg(f"remaining_power"))
            return False
        if get_distance(self.x, self.y, task.creator.x, task.creator.y) > self.radius:
            # print(blue_bg(f"distance"))
            return False
        return True
    
    def get_transmission_time(self ,task ,fixed_fog_nodes) -> float:
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

                    #TODO check this after changing zone manager location
                    if closest_fn.x == 4214.90 and closest_fn.y == 1932.26:
                        return (task.dataSize / dataRate) + (
                                task.dataSize / Config.CloudConfig.CLOUD_BANDWIDTH)
                        # print(blue_bg(f"executor: {task.executor.id}::: delay: {(task.dataSize / dataRate) + (task.dataSize / Config.CloudConfig.CLOUD_BANDWIDTH)}, dataRate: {dataRate}"))
                    else:
                        return (task.dataSize / dataRate) + 2 * (
                                task.dataSize / Config.CloudConfig.CLOUD_BANDWIDTH)


    #TODO for set running task should check current-time >= start-time remove runnig task and use queue
    def assign_task(self, task, current_time: float, fixed_fog_nodes) -> None:
        """Offload a task in the current node."""
        # todo: should change power concept
        # todo: add multicore and Worst Fit Decreasing assigning
        # 1. Initialize task execution parameters
        # Calculate the TOTAL discrete time steps this task needs to complete
        self.tasks.append(task)
        task.total_exec_time = findExecTimeInEachKindOfNode(task)
        task.remaining_time = task.total_exec_time
        task.executor = self

        delay = self.get_transmission_time(task,fixed_fog_nodes)
        task.start_time = current_time + delay
        # 2. Worst Fit Selection: Find the core with the minimum current load
        least_loaded_core_idx = self.core_loads.index(min(self.core_loads))
        self.core_loads[least_loaded_core_idx] += task.total_exec_time
        heapq.heappush(self.cores[least_loaded_core_idx], (task.deadline, task.release_time, task))


    # In your NodeABC class


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
            core_heap = self.cores[i]  # Min-heap sorted by (deadline, ..., task)

            # Temporary store for not-yet-ready tasks (start_time > current_time)
            temp_unready_tasks = []

            # Keep executing tasks until tick time runs out
            while remaining_work_this_tick > 0:
                if not core_heap:
                    break  # No tasks to execute

                # Pop the task with earliest deadline
                deadline, _, task = heapq.heappop(core_heap)

                # Skip tasks not yet started
                if task.start_time > current_time:
                    temp_unready_tasks.append((deadline, _, task))
                    continue

                # Mark the actual start time once
                if not hasattr(task, 'actual_start_time'):
                    task.actual_start_time = current_time

                # Calculate how much work to do
                work_to_do = min(remaining_work_this_tick, task.remaining_time)
                task.remaining_time -= work_to_do
                self.core_loads[i] -= work_to_do
                remaining_work_this_tick -= work_to_do

                # If the task finished, record it
                if task.remaining_time <= 0:
                    task.finish_time = current_time
                    finished_tasks_this_step.append(task)
                    # Continue to use remaining tick power if available
                    continue
                else:
                    # Task still needs work — push it back
                    heapq.heappush(core_heap, (deadline, _, task))
                    # Tick fully consumed (no more time to run next task)
                    break

            # Reinsert unready tasks
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
    def used_power_limit(self) -> float:
        """The maximum power that the node can use to execute its tasks."""
        raise NotImplementedError


@dataclass
class CriticalUserNode(NodeABC):
    """
    Represents the critical-level processor that co-exists inside a UserNode.
    It inherits movement (x, y) from its parent but has its own task queue and resources.
    """
    parent_node: 'MobileNodeABC' = field(init=False)
    id: str = field(init=False)
    x: float = field(init=False)
    y: float = field(init=False)

    def __post_init__(self):
        """
        Initializes the critical processor's default properties from the config.
        Note: Properties depending on the parent (like id, radius)
        must be set in 'set_parent_node'.
        """
        self.power = Config.CriticalUserNodeConfig.DEFAULT_COMPUTATION_POWER
        self.frequency = Config.CriticalUserNodeConfig.USER_NODE_FREQUENCY
        self.remaining_power = self.power
        self.tasks = deque()
        self.periodic_jobs_active = []   
        self.finished_tasks = deque()
        self.last_tbs_deadline = 0.0       # for TBS
        self.next_periodic_release = {p: 0.0 for (p, _, _) in PERIODIC_TASKS}

    def set_parent_node(self, parent: 'MobileNodeABC'):
        """
        Finalizes initialization by linking this node to its parent UserNode.
        This must be called by the parent immediately after creation.
        """
        self.parent_node = parent
        self.id = f"{self.parent_node.id}_critical"
        self.x = self.parent_node.x
        self.y = self.parent_node.y
        self.radius = self.parent_node.radius  # Inherit radius from parent

    def release_periodic_jobs(self, current_time):
        """Release new periodic jobs at their release times."""
        new_jobs = []
        for idx, (period, data_range, cycles_range) in enumerate(PERIODIC_TASKS, start=1):
            # check if it's time for release
            if current_time >= self.next_periodic_release[period] - 1e-9:
                data_kb = random.uniform(*data_range)
                cycles_per_bit = random.uniform(*cycles_range)
                bits = data_kb * 1024
                exec_time = (bits * cycles_per_bit) / self.frequency
                deadline = current_time + period
                task = Task(
                    release_time=current_time,
                    deadline=deadline,
                    exec_time=exec_time,
                    power=0,
                    creator_id=f"{self.id}",
                    dataSize=data_kb,
                    cycles_per_bit=cycles_per_bit,
                    remaining_time=exec_time,
                    start_time=current_time
                )
                new_jobs.append(task)
                self.next_periodic_release[period] += period
        self.periodic_jobs_active.extend(new_jobs)

    def execute_tasks(self, current_time: float, fixed_fog_nodes) -> list:
        """
        Run one time-step of EDF+TBS scheduling.
        If tasks finish early, continue executing others until the timestep ends.
        Returns: list of finished Task objects in this step.
        """
        timestep = 1.0
        finished_tasks_this_step = []

        # 1️⃣ release new periodic jobs if needed
        self.release_periodic_jobs(current_time)

        # 2️⃣ Build ready list (periodic + aperiodic)
        ready_jobs = []
        for task in self.periodic_jobs_active:
            if task.remaining_time > 0:
                heapq.heappush(ready_jobs, (task.deadline, task))

        # 3️⃣ Assign TBS deadlines to new aperiodic tasks
        while self.tasks:
            task = self.tasks.popleft()
            Ck = task.exec_time
            rk = current_time

            # compute Us = 1 - Up (based on current periodic utilization)
            total_util = 0.0
            for (period, data_range, cycles_range) in PERIODIC_TASKS:
                max_data = max(data_range)
                max_cycles = max(cycles_range)
                bits = max_data * 1024
                Ci = (bits * max_cycles) / self.frequency
                total_util += Ci / period
            Us = max(0.1, 1.0 - total_util)

            dk = max(rk, self.last_tbs_deadline) + (Ck / Us)
            self.last_tbs_deadline = dk
            task.deadline = dk
            task.start_time = rk
            heapq.heappush(ready_jobs, (task.deadline, task))

        # 4️⃣ EDF loop — run until timestep exhausted
        time_remaining = timestep
        while time_remaining > 1e-9 and ready_jobs:
            _, running_task = heapq.heappop(ready_jobs)

            run_time = min(running_task.remaining_time, time_remaining)
            running_task.remaining_time -= run_time
            time_remaining -= run_time

            # mark completion or reinsert
            if running_task.remaining_time <= 1e-9:
                running_task.finish_time = current_time + (timestep - time_remaining)
                finished_tasks_this_step.append(running_task)

                if running_task.creator_id.startswith("P"):
                    self.periodic_jobs_active = [
                        j for j in self.periodic_jobs_active if j is not running_task
                    ]
            else:
                # still has remaining time, put back
                heapq.heappush(ready_jobs, (running_task.deadline, running_task))

        # 5️⃣ store finished tasks and return
        self.finished_tasks.extend(finished_tasks_this_step)
        return finished_tasks_this_step

    
    # --- Abstract Method Implementations ---
    @property
    def max_tasks_queue_len(self) -> int:
        return Config.CriticalUserNodeConfig.MAX_TASK_QUEUE_LEN

    @property
    def layer(self) -> Layer:
        return Layer.CriticalUser

    @property
    def used_power_limit(self) -> float:
        return Config.CriticalUserNodeConfig.POWER_LIMIT

@dataclass
class MobileNodeABC(NodeABC, abc.ABC):
    """
    Represents any computational resource in the system which can move from one location to another. Mobile nodes may
    belong to different layers, such as User or Fog.
    """

    speed: float = 0
    angle: float = 0

    # This will hold the internal critical processor
    critical_processor: CriticalUserNode = field(init=False)

    def __post_init__(self):
        """
        Initializes the normal processor and then creates and links
        its internal critical processor.
        """
        # 1. Initialize self (normal processor)
        self.power = Config.UserNodeConfig.DEFAULT_COMPUTATION_POWER
        self.frequency = Config.UserNodeConfig.USER_NODE_FREQUENCY
        self.radius = Config.MobileFogNodeConfig.DEFAULT_RADIUS
        self.remaining_power = self.power
        self.tasks = deque()
        self.finished_tasks = deque()

        # 2. Create and link the critical processor
        self.critical_processor = CriticalUserNode()
        self.critical_processor.set_parent_node(self)

    # --- Overridden Task Execution Method ---
    def execute_tasks(self, current_time: float, fixed_fog_nodes) -> list:
        """
        Executes tasks for BOTH the normal queue and the critical queue.
        This is called by the simulator.
        """
        # 1. Execute tasks from the normal queue (by calling the parent's method)
        finished_normal_tasks = super().execute_tasks(current_time, fixed_fog_nodes)

        # 2. Execute tasks from the critical processor's queue
        # todo: should change the execution to mehrshad's code version
        finished_critical_tasks = self.critical_processor.execute_tasks(current_time, fixed_fog_nodes)

        # 3. Return the combined list of all finished tasks
        return finished_normal_tasks + finished_critical_tasks