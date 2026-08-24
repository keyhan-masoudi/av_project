from typing import Unpack
from controllers.zone_managers.base import ZoneManagerABC, ZoneManagerUpdate
from models.node.fog import FogLayerABC
from models.task import Task

class DeepRLZoneManagerMAPPO(ZoneManagerABC):
    def __init__(self, zone):
        super().__init__(zone)
        self.agent_id = -1
        self.controller = None

    def can_offload_task(self, task: Task) -> bool:
        all_fog_nodes = list(self.fixed_fog_nodes.values()) + list(self.mobile_fog_nodes.values())
        return any(node.can_offload_task(task) for node in all_fog_nodes)

    def assign_task(self, task: Task) -> FogLayerABC:
        pass

    def update(self, **kwargs: Unpack[ZoneManagerUpdate]):
        pass