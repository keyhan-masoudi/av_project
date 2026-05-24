from __future__ import annotations

import abc
import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Dict
import xml.etree.ElementTree as ET

import numpy as np
import random

from config import Config
from controllers.metric import MetricsController
from models.base import ModelBaseABC
from utils.enums import Layer
from utils.distance import get_distance
from models.task import Task


def blue_bg(text):
    return f"\033[44m{text}\033[0m"


def green_bg(text):
    return f"\033[42m{text}\033[0m"

# todo should change this
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
    execution_log = None

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

    def get_best_queue_length(self) -> float:
        """Returns: The length of the shortest queue among all the blinds of this node"""
        if not hasattr(self, 'cores') or not self.cores:
            return 0.0
        return float(min(len(core) for core in self.cores))

    def get_avg_queue_length(self) -> float:
        """Returns: Average queue length at this node"""
        if not hasattr(self, 'cores') or not self.cores:
            return 0.0
        total_tasks = sum(len(core) for core in self.cores)
        return float(total_tasks) / max(1, self.num_cores)

    def get_idle_cores_count(self) -> float:
        """Returns: Number of cores which are IDLE"""
        if not hasattr(self, 'cores') or not self.cores:
            return 0.0
        return float(sum(1 for core in self.cores if len(core) == 0))

    def get_idle_capable_cores_count(self, task) -> float:
        """Returns: The number of idle cores that have the processing power for this task"""
        if not self.can_offload_task(task):
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

    # todo: should change it
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
        # ایجاد یک لیست برای ذخیره تاریخچه اجرا جهت رسم گانت چارت
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

                # -------------- اضافه کردن لاگ اجرا برای گانت چارت --------------
                slice_start = current_time + (WORK_PER_TICK - remaining_work_this_tick)
                self.execution_log.append({
                    'core': i,
                    'task_id': task.id,
                    'start': slice_start,
                    'duration': work_to_do,
                    'is_hard': getattr(task, 'is_hard', False)
                })
                # ----------------------------------------------------------------

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


# @dataclass
# class CriticalUserNode(NodeABC):
#     """
#     Represents the critical-level processor that co-exists inside a UserNode.
#     It inherits movement (x, y) from its parent but has its own task queue and resources.
#     """
#     parent_node: 'MobileNodeABC' = field(init=False)
#     id: str = field(init=False)
#     x: float = field(init=False)
#     y: float = field(init=False)
#
#     def __post_init__(self):
#         """
#         Initializes the critical processor's default properties from the config.
#         Note: Properties depending on the parent (like id, radius)
#         must be set in 'set_parent_node'.
#         """
#         self.periodic_counter = 0
#         self.power = Config.CriticalUserNodeConfig.DEFAULT_COMPUTATION_POWER
#         self.frequency = Config.CriticalUserNodeConfig.USER_NODE_FREQUENCY
#         self.remaining_power = self.power
#         self.tasks = deque()
#         self.periodic_jobs_active = []
#         self.finished_tasks = deque()
#         self.last_tbs_deadline = 0.0  # for TBS
#         self.next_periodic_release = {p: 0.0 for (p, _, _) in PERIODIC_TASKS}
#
#     def set_parent_node(self, parent: 'MobileNodeABC'):
#         """
#         Finalizes initialization by linking this node to its parent UserNode.
#         This must be called by the parent immediately after creation.
#         """
#         self.parent_node = parent
#         self.id = f"{self.parent_node.id}_critical"
#         self.x = self.parent_node.x
#         self.y = self.parent_node.y
#         self.radius = self.parent_node.radius
#
#     def release_periodic_jobs(self, current_time):
#         """Release new periodic jobs at their release times."""
#         new_jobs = []
#         for idx, (period, data_range, cycles_range) in enumerate(PERIODIC_TASKS, start=1):
#             # check if it's time for release
#             if current_time >= self.next_periodic_release[period] - 1e-9:
#                 data_kb = random.uniform(*data_range)
#                 cycles_per_bit = random.uniform(*cycles_range)
#                 bits = data_kb * 1024
#                 exec_time = (bits * cycles_per_bit) / self.frequency
#                 deadline = current_time + period
#                 self.periodic_counter += 1
#                 task_id = f"P_{self.id}_{self.periodic_counter}"
#                 task = Task(
#                     id=task_id,
#                     release_time=current_time,
#                     deadline=deadline,
#                     exec_time=exec_time,
#                     power=0,
#                     creator_id=f"#{self.id}",
#                     dataSize=data_kb,
#                     cycles_per_bit=cycles_per_bit,
#                     remaining_time=exec_time,
#                     start_time=current_time
#                 )
#                 new_jobs.append(task)
#                 self.next_periodic_release[period] += period
#         self.periodic_jobs_active.extend(new_jobs)
#
#     def execute_tasks(self, current_time: float, fixed_fog_nodes) -> list:
#         """
#         Run one time-step of EDF+TBS scheduling.
#         If tasks finish early, continue executing others until the timestep ends.
#         Returns: list of finished Task objects in this step.
#         """
#         timestep = 1.0
#         finished_tasks_this_step = []
#
#         # 1️⃣ release new periodic jobs if needed
#         self.release_periodic_jobs(current_time)
#
#         # 2️⃣ Build ready list (periodic + aperiodic)
#         ready_jobs = []
#         for task in self.periodic_jobs_active:
#             if task.remaining_time > 0:
#                 heapq.heappush(ready_jobs, (task.deadline, task))
#
#         # 3️⃣ Assign TBS deadlines to new aperiodic tasks
#         while self.tasks:
#             task = self.tasks.popleft()
#             Ck = task.exec_time
#             rk = current_time
#
#             # compute Us = 1 - Up (based on current periodic utilization)
#             total_util = 0.0
#             for (period, data_range, cycles_range) in PERIODIC_TASKS:
#                 max_data = max(data_range)
#                 max_cycles = max(cycles_range)
#                 bits = max_data * 1024
#                 Ci = (bits * max_cycles) / self.frequency
#                 total_util += Ci / period
#             Us = max(0.1, 1.0 - total_util)
#
#             dk = max(rk, self.last_tbs_deadline) + (Ck / Us)
#             self.last_tbs_deadline = dk
#             task.deadline = dk
#             task.start_time = rk
#             heapq.heappush(ready_jobs, (task.deadline, task))
#
#         # 4️⃣ EDF loop — run until timestep exhausted
#         time_remaining = timestep
#         while time_remaining > 1e-9 and ready_jobs:
#             _, running_task = heapq.heappop(ready_jobs)
#
#             run_time = min(running_task.remaining_time, time_remaining)
#             running_task.remaining_time -= run_time
#             time_remaining -= run_time
#
#             # mark completion or reinsert
#             if running_task.remaining_time <= 1e-9:
#                 running_task.finish_time = current_time + (timestep - time_remaining)
#                 finished_tasks_this_step.append(running_task)
#
#                 if running_task.creator_id.startswith("#"):
#                     self.periodic_jobs_active = [
#                         j for j in self.periodic_jobs_active if j is not running_task
#                     ]
#                 else:
#                     self.tasks = [  # TODO
#                         j for j in self.tasks if j is not running_task
#                     ]
#             else:
#                 # still has remaining time, put back
#                 heapq.heappush(ready_jobs, (running_task.deadline, running_task))
#
#         # 5️⃣ store finished tasks and return
#         self.finished_tasks.extend(finished_tasks_this_step)
#         return finished_tasks_this_step
#
#     # --- Abstract Method Implementations ---
#     @property
#     def max_tasks_queue_len(self) -> int:
#         return Config.CriticalUserNodeConfig.MAX_TASK_QUEUE_LEN
#
#     @property
#     def layer(self) -> Layer:
#         return Layer.CriticalUser
#
#     @property
#     def num_cores(self) -> int:
#         return Config.CriticalUserNodeConfig.NUM_CORE


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

        # محاسبه زمان اجرای واقعی
        exec_time_calculated = findExecTimeInEachKindOfNode(task)
        task.total_exec_time = exec_time_calculated if exec_time_calculated > 0 else task.exec_time
        task.remaining_time = task.total_exec_time
        task.start_time = current_time
        task.is_hard = True
        self.local_hard_tasks.append(task)

        # -------------------------------------------------------------
        # پیاده‌سازی Worst Fit Decreasing / Worst Fit بر اساس Utilization
        # -------------------------------------------------------------
        # محاسبه دوره تناوب (Period) فرض شده از روی فاصله Release Time و Deadline
        period = task.deadline - task.release_time
        # محاسبه Utilization این تسک هارد
        task_utilization = task.total_exec_time / period if period > 0 else task.total_exec_time

        # پیدا کردن کوری که کمترین مقدار Utilization هارد (core_Up) را دارد (الگوریتم Worst Fit)
        best_core_idx = self.core_Up.index(min(self.core_Up))

        # آپدیت کردن Utilization پریودیک (Up) برای کور انتخاب شده
        self.core_Up[best_core_idx] += task_utilization

        # آپدیت کردن پهنای باند سرور آپریودیک (Us) برای همان کور
        # برای جلوگیری از تقسیم بر صفر در فرمول TBS، مینیمم Us را 0.01 در نظر می‌گیریم
        self.core_Us[best_core_idx] = max(0.01, 1.0 - self.core_Up[best_core_idx])

        # آپدیت کردن بار کلی کور (برای لاگ‌ها)
        self.core_loads[best_core_idx] += task.total_exec_time

        # پوش کردن تسک هارد داخل صف (Heap) همان کور بر اساس ددلاین
        heapq.heappush(self.cores[best_core_idx], (task.deadline, task.release_time, task))

    def assign_task(self, task, current_time: float, fixed_fog_nodes=None) -> None:
        """آف‌لود وظایف آپریودیک با استفاده از فرمول TBS برای گره‌های سیار"""
        self.tasks.append(task)
        task.executor = self
        task.total_exec_time = findExecTimeInEachKindOfNode(task)
        task.remaining_time = task.total_exec_time

        delay = 0
        if hasattr(self, 'get_transmission_time') and fixed_fog_nodes is not None:
            delay = self.get_transmission_time(task, fixed_fog_nodes)

        task.start_time = current_time + delay
        rk = task.start_time

        best_core_idx = 0
        min_prospective_deadline = float('inf')

        for i in range(self.num_cores):
            Us = self.core_Us[i]  # این مقدار حالا توسط تسک‌های هارد آپدیت شده است
            last_dl = self.last_tbs_deadline[i]
            Ck = task.total_exec_time

            # فرمول عکس: dk = max(rk, last_deadline) + (Ck / Us)
            prospective_dk = max(rk, last_dl) + (Ck / Us)

            # انتخاب کوری که در نهایت بهترین ددلاین (کمترین ددلاین ممکن) را به ما می‌دهد
            if prospective_dk < min_prospective_deadline:
                min_prospective_deadline = prospective_dk
                best_core_idx = i

        # # ست کردن ددلاین نهایی بر اساس فرمول
        # task.deadline = min_prospective_deadline

        # آپدیت کردن آخرین ددلاین محاسبه شده برای این کور جهت استفاده در تسک‌های بعدی
        self.last_tbs_deadline[best_core_idx] = min_prospective_deadline

        self.core_loads[best_core_idx] += task.total_exec_time

        # قرار دادن تسک سافت در صف اولویت
        heapq.heappush(self.cores[best_core_idx], (min_prospective_deadline, task.release_time, task))

    def execute_tasks(self, current_time: float, fixed_fog_nodes) -> list:
        """
            Executes tasks on all cores for one time step using Preemptive EDF.
            Relies on a heap to manage task priority by deadline.
            - Always runs the task with the earliest deadline.
            - Skips tasks whose start_time > current_time.
            - If a task finishes before the tick ends, continues with the next ready task.
        """
        # ایجاد یک لیست برای ذخیره تاریخچه اجرا جهت رسم گانت چارت
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

                # -------------- اضافه کردن لاگ اجرا برای گانت چارت --------------
                slice_start = current_time + (WORK_PER_TICK - remaining_work_this_tick)
                self.execution_log.append({
                    'core': i,
                    'task_id': task.id,
                    'start': slice_start,
                    'duration': work_to_do,
                    'is_hard': getattr(task, 'is_hard', False)
                })
                # ----------------------------------------------------------------

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
