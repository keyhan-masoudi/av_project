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
    def used_power_limit(self) -> float:
        return Config.FixedFogNodeConfig.POWER_LIMIT


class MobileFogNode(FogLayerABC, MobileNodeABC):
    """Represents a fog node that can move around in the system, migrating from one zone to another."""

    @property
    def type(self) -> FogType:
        return FogType.MOBILE

    @property
    def max_tasks_queue_len(self) -> int:
        return Config.MobileFogNodeConfig.MAX_TASK_QUEUE_LEN

    @property
    def used_power_limit(self) -> float:
        return Config.MobileFogNodeConfig.POWER_LIMIT
