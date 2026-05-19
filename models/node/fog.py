import abc
from dataclasses import dataclass

from config import Config
from models.node.base import MobileNodeABC, NodeABC
from utils.enums import FogType, Layer


class FogLayerABC(NodeABC, abc.ABC):
    """
    An abstract representation of fog layer that is responsible for providing computational resources to Users faster
    than Cloud layer.
    """
    @property
    def layer(self) -> Layer:
        return Layer.FOG

    @property
    @abc.abstractmethod
    def type(self) -> FogType:
        raise NotImplementedError


@dataclass
class FixedFogNode(FogLayerABC):
    """Represents a fog node in the system which is located in fixed coordination."""
    @property
    def type(self) -> FogType:
        return FogType.FIXED

    @property
    def max_tasks_queue_len(self) -> int:
        return Config.FixedFogNodeConfig.MAX_TASK_QUEUE_LEN

    @property
    def num_cores(self) -> int:
        return Config.FixedFogNodeConfig.NUM_CORE


class MobileFogNode(FogLayerABC, MobileNodeABC):
    """Represents a fog node that can move around in the system, migrating from one zone to another."""

    def __post_init__(self):
        super().__post_init__()
        self.power = Config.MobileFogNodeConfig.DEFAULT_COMPUTATION_POWER
        self.frequency = Config.MobileFogNodeConfig.MOBILE_NODE_FREQUENCY
        self.remaining_power = self.power

    @property
    def type(self) -> FogType:
        return FogType.MOBILE

    @property
    def max_tasks_queue_len(self) -> int:
        return Config.MobileFogNodeConfig.MAX_TASK_QUEUE_LEN

    @property
    def num_cores(self) -> int:
        return Config.MobileFogNodeConfig.NUM_CORE

    def execute_tasks(self, current_time: float, fixed_fog_nodes) -> list:
        """Run hard tasks locally, then execute any other tasks on this vehicle."""
        finished_hard = self._execute_local_hard_tasks(current_time, float(self.num_cores))
        finished_other = super().execute_tasks(current_time, fixed_fog_nodes)
        return finished_hard + finished_other
