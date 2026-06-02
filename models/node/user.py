from config import Config
from models.node.base import MobileNodeABC
from utils.enums import Layer


class UserNode(MobileNodeABC):
    """User vehicle with one local multicore processor (TBS + EDF for hard and soft)."""

    def __post_init__(self):
        super().__post_init__()
        self.power = Config.UserNodeConfig.DEFAULT_COMPUTATION_POWER
        self.frequency = Config.UserNodeConfig.USER_NODE_FREQUENCY
        self.remaining_power = self.power

    @property
    def max_tasks_queue_len(self) -> int:
        return Config.UserNodeConfig.MAX_TASK_QUEUE_LEN

    @property
    def layer(self) -> Layer:
        return Layer.USER

    @property
    def num_cores(self) -> int:
        return Config.UserNodeConfig.NUM_CORE
