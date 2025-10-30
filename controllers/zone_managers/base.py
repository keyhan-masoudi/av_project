import abc
from typing import List, TypedDict, Unpack, Tuple

from models.node.fog import FixedFogNode, FogLayerABC, MobileFogNode
from models.task import Task
from models.zone import Zone


class ZoneManagerUpdate(TypedDict):
    current_time: float
    all_zone_managers: List['ZoneManagerABC']
    tasks: List[Task]
    current_task: Task


class ZoneManagerABC(abc.ABC):
    def __init__(self, zone: Zone):
        self.zone: Zone = zone
        self.fixed_fog_nodes: dict[str, FixedFogNode] = {}
        self.mobile_fog_nodes: dict[str, MobileFogNode] = {}

    def add_fixed_fog_nodes(self, fixed_fog_nodes: List[FixedFogNode]):
        for node in fixed_fog_nodes:
            self.fixed_fog_nodes[node.id] = node

    def set_mobile_fog_nodes(self, mobile_fog_nodes: List[MobileFogNode]):
        data = {}
        for node in mobile_fog_nodes:
            data[node.id] = node

        self.mobile_fog_nodes = data

    def offload_task(self, task: Task, current_time: float) -> FogLayerABC:
        assigned_node = self.assign_task(task)
        # todo: don't forget to check following code and add it in our new version !
        # assigned_node.assign_task(task, current_time)
        return assigned_node

    @abc.abstractmethod
    def can_offload_task(self, task: Task) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def assign_task(self, task: Task) -> FogLayerABC:
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, **kwargs: Unpack[ZoneManagerUpdate]):
        raise NotImplementedError
