from typing import Unpack

from controllers.zone_managers.base import ZoneManagerABC, ZoneManagerUpdate
from models.node.fog import FogLayerABC
from models.task import Task


class OnlyCloudZoneManager(ZoneManagerABC):
    def assign_task(self, task: Task) -> FogLayerABC:
        return None

    def can_offload_task(self, task: Task) -> bool:
        return False

    def update(self, **kwargs: Unpack[ZoneManagerUpdate]):
        pass
