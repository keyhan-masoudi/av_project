from config import Config
from models.node.base import MobileNodeABC
from utils.enums import Layer


class UserNode(MobileNodeABC):
    """Represents users in the system which can move from one location to another."""

    @property
    def max_tasks_queue_len(self) -> int:
        return Config.UserNodeConfig.MAX_TASK_QUEUE_LEN

    @property
    def layer(self) -> Layer:
        return Layer.USER

    @property
    def used_power_limit(self) -> float:
        return Config.UserNodeConfig.POWER_LIMIT
