from dataclasses import dataclass

import numpy as np

from config import Config
from models.base import ModelBaseABC
from models.node.base import MobileNodeABC, NodeABC
from utils.enums import Layer


@dataclass
class Task(ModelBaseABC):
    """Represents tasks that each mobile node can produce over time."""
    release_time: float
    deadline: float
    exec_time: float  # The amount of time that this task required to execute.
    power: float  # The amount of power unit that this tasks consumes while executing.
    creator_id: str  # Thd id of the node who created the task.
    dataSize: float

    start_time: float = 0  # The time that this task was offloaded to a node (either local or external).
    finish_time: float = 0  # The time that this task was finished in the offloaded node.
    real_exec_time_base: float = 0
    SNR: float = 0

    creator: MobileNodeABC = None
    executor: NodeABC = None

    def real_exec_time(self, executor=None) -> float:
        # print(f"self.executor : {self.executor}\nexecutor : {executor}")
        taskExecutor = self.executor
        if taskExecutor is None:
            taskExecutor = executor
        if self.creator.id == taskExecutor.id and self.creator.layer == Layer.USER:
            return self.exec_time * Config.UserNodeConfig.LOCAL_EXECUTE_TIME_OVERHEAD
        return self.exec_time

    @property
    def has_migrated(self) -> bool:
        if self.executor.radius > self.get_creator_and_executor_distance():
            return False
        return True

    def get_creator_and_executor_distance(self, executor=None) -> float:
        taskExecutor = self.executor
        if taskExecutor is None:
            taskExecutor = executor
        return np.sqrt((self.creator.x - taskExecutor.x) ** 2 + (self.creator.y - taskExecutor.y) ** 2)

    @property
    def is_deadline_missed(self) -> bool:
        return self.deadline < self.finish_time

    @property
    def is_completed(self) -> bool:
        return self.finish_time > 0
