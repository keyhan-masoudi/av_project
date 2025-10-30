import random

from typing import Dict, List, Unpack

from config import Config
from controllers.zone_managers.base import ZoneManagerABC, ZoneManagerUpdate
from models.node.fog import FogLayerABC
from models.task import Task


class OnlyFogZoneManager(ZoneManagerABC):
    def can_offload_task(self, task: Task) -> bool:
        merged_fog_nodes: Dict[str, FogLayerABC] = {**self.fixed_fog_nodes, **self.mobile_fog_nodes}

        possible_fog_nodes: List[FogLayerABC] = []
        for fog_id, fog in merged_fog_nodes.items():
            if fog.can_offload_task(task):
                possible_fog_nodes.append(fog)

        if len(possible_fog_nodes) == 0:
            return False
        self.__possible_fog_nodes = possible_fog_nodes
        return True

    def update(self, **kwargs: Unpack[ZoneManagerUpdate]):
        pass

    def assign_task(self, task: Task) -> FogLayerABC:
        return random.choice(self.__possible_fog_nodes)
