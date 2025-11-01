from config import Config
from models.node.base import NodeABC
from utils.enums import Layer


class CloudNode(NodeABC):
    """Represents a cloud layer in the system which has unlimited amount of computational resource."""

    power = Config.CloudConfig.DEFAULT_COMPUTATION_POWER
    remaining_power = Config.CloudConfig.DEFAULT_COMPUTATION_POWER


    @property
    def max_tasks_queue_len(self) -> int:
        return Config.CloudConfig.MAX_TASK_QUEUE_LEN

    @property
    def used_power_limit(self) -> float:
        return Config.CloudConfig.POWER_LIMIT

    @property
    def layer(self) -> Layer:
        return Layer.CLOUD

