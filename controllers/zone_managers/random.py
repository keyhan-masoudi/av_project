import random
from typing import Dict, List, Unpack, Any

from config import Config
from controllers.zone_managers.base import ZoneManagerABC, ZoneManagerUpdate
from models.node.fog import FogLayerABC
from models.task import Task


class RandomZoneManager(ZoneManagerABC):
    def __init__(self, zone):
        super().__init__(zone)
        self.simulator = None

    def set_simulator(self, simulator):
        self.simulator = simulator

    def can_offload_task(self, task: Task) -> bool:
        merged_fog_nodes: Dict[str, FogLayerABC] = {**self.fixed_fog_nodes, **self.mobile_fog_nodes}

        possible_nodes = []

        if task.creator.can_offload_task(task):
            possible_nodes.append(task.creator)

        for fog_id, fog in merged_fog_nodes.items():
            if fog.can_offload_task(task):
                possible_nodes.append(fog)

        if self.simulator.cloud_node:
            if self.simulator.cloud_node.can_offload_task(task):
                possible_nodes.append(self.simulator.cloud_node)

        if len(possible_nodes) == 0:
            return False

        self.__possible_nodes = possible_nodes
        return True

    def update(self, **kwargs: Unpack[ZoneManagerUpdate]):
        pass

    def assign_task(self, task: Task) -> Any:
        return random.choice(self.__possible_nodes)
