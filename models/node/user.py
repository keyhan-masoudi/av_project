import heapq
from collections import deque

from config import Config
from models.node.base import MobileNodeABC
from utils.enums import Layer


class UserNode(MobileNodeABC):
    """User vehicle with one local multicore processor running TBS + EDF."""

    def __post_init__(self):
        super().__post_init__()
        self.power = Config.UserNodeConfig.DEFAULT_COMPUTATION_POWER
        self.frequency = Config.UserNodeConfig.USER_NODE_FREQUENCY
        self.remaining_power = self.power
        self.last_tbs_deadline = 0.0

    @property
    def max_tasks_queue_len(self) -> int:
        return Config.UserNodeConfig.MAX_TASK_QUEUE_LEN

    @property
    def layer(self) -> Layer:
        return Layer.USER

    @property
    def num_cores(self) -> int:
        return Config.UserNodeConfig.NUM_CORE

    def assign_task(self, task, current_time: float, fixed_fog_nodes) -> None:
        """Queue a locally executed soft task; TBS deadline is set in execute_tasks."""
        if task.is_hard:
            self.assign_local_hard_task(task, current_time)
            return

        task.creator = self
        task.executor = self
        task.release_time = current_time
        self.tasks.append(task)
        task.total_exec_time = task.real_exec_time(executor=self)
        task.remaining_time = task.total_exec_time
        task.start_time = current_time

    def _hard_task_utilization(self) -> float:
        """Worst-case utilization of hard periodic tasks for TBS."""
        total_util = 0.0
        divisor = Config.UserNodeConfig.HARD_TASK_EXEC_TIME_DIVISOR
        for spec in Config.UserNodeConfig.HARD_TASK_SPECS:
            worst_exec = (spec["size_max"] * spec["cycles_max"]) / (self.frequency * divisor)
            total_util += worst_exec / spec["period"]
        return total_util

    def execute_tasks(self, current_time: float, fixed_fog_nodes) -> list:
        """Run one timestep of TBS + EDF on hard and local soft tasks."""
        timestep = float(self.num_cores)
        finished_tasks_this_step = []
        ready_jobs = []

        for task in self.local_hard_tasks:
            if task.remaining_time > 0 and task.start_time <= current_time:
                heapq.heappush(ready_jobs, (task.deadline, task))

        slack_util = max(0.1, 1.0 - self._hard_task_utilization())
        while self.tasks:
            task = self.tasks.popleft()
            if task.executor is not self:
                continue
            release_time = current_time
            self.last_tbs_deadline = max(
                release_time,
                self.last_tbs_deadline,
            ) + (task.exec_time / slack_util)
            task.deadline = self.last_tbs_deadline
            task.start_time = release_time
            if task.remaining_time <= 0:
                task.remaining_time = task.total_exec_time
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
                heapq.heappush(ready_jobs, (running_task.deadline, running_task))

        self.finished_tasks.extend(finished_tasks_this_step)
        return finished_tasks_this_step
