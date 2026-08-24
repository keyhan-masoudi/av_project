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
from models import task
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
        task.start_time = math.ceil(current_time + delay)
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
    last_itbs_deadline: List[float] = field(init=False)
    periodic_allocation: List[List[dict]] = field(init=False)
    local_hard_tasks: list = field(init=False, default_factory=list)
    registered_hard_task_types: set = field(init=False, default_factory=set)
    hard_task_specs: List[List[dict]] = field(init=False)
    hard_task_phases: Dict[int, float] = field(init=False, default_factory=dict)

    hard_cores: List[List[tuple]] = field(init=False)
    soft_cores: List[List[tuple]] = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.core_Up = [0.0] * self.num_cores
        self.core_Us = [1.0] * self.num_cores
        self.last_tbs_deadline = [0.0] * self.num_cores
        self.last_itbs_deadline = [0.0] * self.num_cores
        self.periodic_allocation = [[] for _ in range(self.num_cores)]
        self.hard_task_specs = [[] for _ in range(self.num_cores)]

        self.hard_cores = [[] for _ in range(self.num_cores)]
        self.soft_cores = [[] for _ in range(self.num_cores)]

        import json
        import os

        # The configured project root still points to the original Windows
        # checkout in some experiments. Resolve the generated parameters from
        # the repository itself before using the legacy fallback.
        repository_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        json_candidates = [
            os.path.join(Config.VehiclesTraffic.PROJECT_ROOT, "data", "hard_task_parameters_uunifast.json"),
            os.path.join(repository_root, "data", "hard_task_parameters_uunifast.json"),
            os.path.join("data", "hard_task_parameters_uunifast.json"),
            "hard_tasks.json",
        ]
        json_path = next((path for path in json_candidates if os.path.isfile(path)), None)

        if json_path is not None:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for type_index, spec in enumerate(data.get("tasks", [])):
                    core_idx = spec.get("core", -1)
                    if 0 <= core_idx < self.num_cores:
                        self.core_Up[core_idx] += spec.get("utilization", 0.0)
                        self.hard_task_specs[core_idx].append({
                            "type_index": type_index,
                            "period": float(spec["period"]),
                            "wcet": float(spec.get("wcet", 0.0)),
                        })
            except Exception as e:
                print(blue_bg(f"Warning: Failed to load JSON: {e}"))
        else:
            print(blue_bg("Warning: WFD JSON not found. ITBS will use default bandwidth."))

        # Calculate the per-core server bandwidth from offline worst-case load.
        for i in range(self.num_cores):
            self.core_Us[i] = max(0.01, 1.0 - self.core_Up[i])

    def assign_local_hard_task(self, task, current_time: float) -> None:
        """Register a hard task on its offline WFD-assigned core (from task.core)."""
        if task.core is None:
            raise ValueError(f"Hard task {task.id} is missing offline core assignment")

        core_idx = int(task.core)
        if not 0 <= core_idx < self.num_cores:
            raise ValueError(
                f"Hard task {task.id} core {core_idx} out of range for {self.num_cores} cores"
            )

        task.creator = self
        task.executor = self
        task.creator_id = self.id
        task.release_time = current_time

        exec_time_calculated = findExecTimeInEachKindOfNode(task)
        base_exec_time = exec_time_calculated if exec_time_calculated > 0 else task.exec_time
        task.is_hard = True
        self.local_hard_tasks.append(task)

        # All instances of a generated hard-task type share this first-release
        # phase.  ITBS uses it with the offline period/WCET to include future
        # periodic releases in its EDF finishing-time calculation.
        if task.type_index is not None and task.type_index not in self.registered_hard_task_types:
            type_index = int(task.type_index)
            self.registered_hard_task_types.add(type_index)
            self.hard_task_phases[type_index] = current_time

        if Config.SimulatorConfig.BASELINE_PARALLEL_FREQUENCY:
            # Scale execution time up to simulate frequency reduction for hard tasks
            task.total_exec_time = base_exec_time / Config.SimulatorConfig.HARD_TASKS_FREQ_RATIO
            task.remaining_time = task.total_exec_time
            task.start_time = current_time
            self.core_loads[core_idx] += task.total_exec_time
            heapq.heappush(self.hard_cores[core_idx], (task.deadline, task.release_time, task))
        else:
            # Standard dynamic allocation mode
            task.total_exec_time = base_exec_time
            task.remaining_time = task.total_exec_time
            task.start_time = current_time
            self.core_loads[core_idx] += task.total_exec_time
            heapq.heappush(self.cores[core_idx], (task.deadline, task.release_time, task))

    @staticmethod
    def _next_periodic_release(phase: float, period: float, time: float) -> float:
        """Return the first release of a periodic task strictly after time."""
        releases_elapsed = math.floor((time - phase) / period) + 1
        return phase + max(0, releases_elapsed) * period

    def _active_periodic_interference(
            self,
            core_idx: int,
            observation_time: float,
            bound_start: float,
            deadline: float,
    ) -> float:
        """Compute I_a(t, d): carry-in periodic work with deadline < d."""
        epsilon = 1e-9

        if abs(bound_start - observation_time) <= epsilon:
            return sum(
                max(0.0, float(task.remaining_time))
                for queued_deadline, _, task in self.cores[core_idx]
                if getattr(task, "is_hard", False)
                and task.start_time <= bound_start + epsilon
                and queued_deadline < deadline - epsilon
            )

        # If the preceding aperiodic request has not completed yet, bound_start
        # is its conservative completion bound in the future. At most one job
        # of each feasible implicit-deadline periodic type can carry into that
        # instant; charging its full WCET is safe and intentionally pessimistic.
        interference = 0.0
        for spec in self.hard_task_specs[core_idx]:
            phase = self.hard_task_phases.get(spec["type_index"])
            period = spec["period"]
            if phase is None or period <= 0.0 or bound_start < phase:
                continue
            release = phase + math.floor((bound_start - phase) / period) * period
            absolute_deadline = release + period
            if (
                    absolute_deadline > bound_start + epsilon
                    and absolute_deadline < deadline - epsilon
            ):
                interference += spec["wcet"]
        return interference

    def _future_periodic_interference(
            self,
            core_idx: int,
            bound_start: float,
            deadline: float,
    ) -> float:
        """Compute I_f(t, d) from the slide's closed-form release count."""
        interference = 0.0
        epsilon = 1e-9

        for spec in self.hard_task_specs[core_idx]:
            phase = self.hard_task_phases.get(spec["type_index"])
            period = spec["period"]
            wcet = spec["wcet"]
            if phase is None or period <= 0.0 or wcet <= 0.0:
                continue

            next_release = self._next_periodic_release(phase, period, bound_start)
            # Number of releases after t whose implicit absolute deadline is
            # strictly before d:
            # max(0, ceil((d - next_r_i(t)) / T_i) - 1).
            count = max(
                0,
                math.ceil(((deadline - next_release) / period) - epsilon) - 1,
            )
            interference += count * wcet

        return interference

    def _improve_tbs_deadline(
            self,
            core_idx: int,
            observation_time: float,
            bound_start: float,
            execution_time: float,
            tbs_deadline: float,
    ) -> float:
        """Apply the slide's bounded-interference Improving TBS recurrence."""
        deadline = tbs_deadline
        epsilon = 1e-9
        max_steps = max(0, Config.SimulatorConfig.ITBS_MAX_REFINEMENT_STEPS)

        for _ in range(max_steps):
            active_interference = self._active_periodic_interference(
                core_idx, observation_time, bound_start, deadline
            )
            future_interference = self._future_periodic_interference(
                core_idx, bound_start, deadline
            )
            finish_bound = (
                bound_start
                + execution_time
                + active_interference
                + future_interference
            )
            improved_deadline = min(deadline, finish_bound)

            if deadline - improved_deadline <= epsilon:
                return deadline
            deadline = improved_deadline

        return deadline

    def assign_task(self, task, current_time: float, fixed_fog_nodes=None) -> None:
        """Assign a soft task using Improved Total Bandwidth Server (ITBS/TB*)."""
        self.tasks.append(task)
        task.executor = self
        base_exec_time = findExecTimeInEachKindOfNode(task)

        delay = self.get_transmission_time(task, fixed_fog_nodes)
        task.start_time = math.ceil(current_time + delay)
        rk = task.start_time

        best_core_idx = 0
        min_prospective_deadline = float('inf')
        prospective_tbs_deadlines = [float('inf')] * self.num_cores

        for i in range(self.num_cores):
            Us = self.core_Us[i]
            last_dl = self.last_tbs_deadline[i]
            Ck = base_exec_time

            tbs_deadline = max(rk, last_dl) + (Ck / Us)
            prospective_tbs_deadlines[i] = tbs_deadline
            # The parallel-frequency baseline models hard and soft work on
            # separate processor shares, not on the unified EDF queue assumed
            # by ITBS. Thus original TBS deadline is kept so baseline comparisons
            # remain semantically unchanged.
            if Config.SimulatorConfig.BASELINE_PARALLEL_FREQUENCY:
                prospective_dk = tbs_deadline
            else:
                prospective_dk = self._improve_tbs_deadline(
                    core_idx=i,
                    observation_time=current_time,
                    bound_start=max(rk, self.last_itbs_deadline[i]),
                    execution_time=Ck,
                    tbs_deadline=tbs_deadline,
                )

            if prospective_dk < min_prospective_deadline:
                min_prospective_deadline = prospective_dk
                best_core_idx = i

        # Keep the original TBS reservation chain separate from its shortened
        # deadline. Reusing the improved deadline here would permit subsequent
        # jobs to consume more than Us and can invalidate the hard-task proof.
        self.last_tbs_deadline[best_core_idx] = prospective_tbs_deadlines[best_core_idx]
        self.last_itbs_deadline[best_core_idx] = min_prospective_deadline
        task.server_deadline = min_prospective_deadline

        if Config.SimulatorConfig.BASELINE_PARALLEL_FREQUENCY:
            # Scale execution time up to simulate frequency reduction for soft tasks
            task.total_exec_time = base_exec_time / (1 - Config.SimulatorConfig.HARD_TASKS_FREQ_RATIO)
            task.remaining_time = task.total_exec_time
            self.core_loads[best_core_idx] += task.total_exec_time
            heapq.heappush(self.soft_cores[best_core_idx], (min_prospective_deadline, task.release_time, task))
        else:
            # Standard dynamic allocation mode
            task.total_exec_time = base_exec_time
            task.remaining_time = task.total_exec_time
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

        is_parallel_baseline = Config.SimulatorConfig.BASELINE_PARALLEL_FREQUENCY

        for i in range(self.num_cores):
            if is_parallel_baseline:
                # -------------------------------------------------------------
                # HARD QUEUE EXECUTION (Runs concurrently with full tick step)
                # -------------------------------------------------------------
                remaining_hard_tick = WORK_PER_TICK
                hard_heap = self.hard_cores[i]
                temp_unready_hard = []

                while remaining_hard_tick > 0 and hard_heap:
                    deadline, rel_time, task = heapq.heappop(hard_heap)

                    if task.start_time > current_time:
                        temp_unready_hard.append((deadline, rel_time, task))
                        continue

                    work_to_do = min(remaining_hard_tick, task.remaining_time)
                    slice_start = current_time + (WORK_PER_TICK - remaining_hard_tick)

                    self.execution_log.append({
                        'core': i,
                        'task_id': task.id,
                        'start': slice_start,
                        'duration': work_to_do,
                        'is_hard': True
                    })

                    task.remaining_time -= work_to_do
                    self.core_loads[i] -= work_to_do
                    remaining_hard_tick -= work_to_do

                    if task.remaining_time <= 0:
                        task.finish_time = current_time + (WORK_PER_TICK - remaining_hard_tick)
                        finished_tasks_this_step.append(task)
                        self.finished_tasks.append(task)
                        if task in getattr(self, 'local_hard_tasks', []):
                            self.local_hard_tasks.remove(task)
                    else:
                        heapq.heappush(hard_heap, (deadline, rel_time, task))
                        break

                for item in temp_unready_hard:
                    heapq.heappush(hard_heap, item)

                # -------------------------------------------------------------
                # SOFT QUEUE EXECUTION (Runs concurrently with full tick step)
                # -------------------------------------------------------------
                remaining_soft_tick = WORK_PER_TICK
                soft_heap = self.soft_cores[i]
                temp_unready_soft = []

                while remaining_soft_tick > 0 and soft_heap:
                    deadline, rel_time, task = heapq.heappop(soft_heap)

                    if task.start_time > current_time:
                        temp_unready_soft.append((deadline, rel_time, task))
                        continue

                    work_to_do = min(remaining_soft_tick, task.remaining_time)
                    slice_start = current_time + (WORK_PER_TICK - remaining_soft_tick)

                    self.execution_log.append({
                        'core': i,
                        'task_id': task.id,
                        'start': slice_start,
                        'duration': work_to_do,
                        'is_hard': False
                    })

                    task.remaining_time -= work_to_do
                    self.core_loads[i] -= work_to_do
                    remaining_soft_tick -= work_to_do

                    if task.remaining_time <= 0:
                        task.finish_time = current_time + (WORK_PER_TICK - remaining_soft_tick)
                        finished_tasks_this_step.append(task)
                        self.finished_tasks.append(task)
                        if task in getattr(self, 'tasks', []):
                            self.tasks.remove(task)
                    else:
                        heapq.heappush(soft_heap, (deadline, rel_time, task))
                        break

                for item in temp_unready_soft:
                    heapq.heappush(soft_heap, item)

            else:
                # -------------------------------------------------------------
                # STANDARD MODE: Unified Priority Queue Execution
                # -------------------------------------------------------------
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

                        if task in getattr(self, 'tasks', []):
                            self.tasks.remove(task)

                        if getattr(task, 'is_hard', False) and task in getattr(self, 'local_hard_tasks', []):
                            self.local_hard_tasks.remove(task)
                    else:
                        heapq.heappush(core_heap, (deadline, rel_time, task))
                        break

                for item in temp_unready_tasks:
                    heapq.heappush(core_heap, item)

        return finished_tasks_this_step
