from typing import Unpack

import numpy as np

from controllers.zone_managers.base import ZoneManagerABC, ZoneManagerUpdate
from models.node.fog import FixedFogNode, FogLayerABC
from models.node.base import MobileNodeABC
from models.task import Task

def yellow_bg(text):
    return f"\033[43m{text}\033[0m"
def blue_bg(text):
    return f"\033[44m{text}\033[0m"
class HeuristicZoneManager(ZoneManagerABC):
    def assign_task(self, task: Task) -> MobileNodeABC | None | FogLayerABC:
        if task.creator.can_offload_task(task):
            return task.creator

        creator = task.creator
        nearest_distance = float('inf')
        nearest_fog_node = None
        for node in self.all_possible_nodes.values():
            next_dis = self.get_next_distance(task, creator, node)
            # print(blue_bg(f"next distance : {next_dis}"))
            if nearest_fog_node is None or next_dis < nearest_distance:
                nearest_fog_node = node
                nearest_distance = next_dis
        return nearest_fog_node

    @staticmethod
    def get_next_distance(task: Task, creator: MobileNodeABC, executor: MobileNodeABC) -> float:
        time = task.exec_time
        creator_next_position_x = (creator.x + creator.speed * time * np.cos(np.deg2rad(creator.angle)))
        creator_next_position_y = (creator.y + creator.speed * time * np.sin(np.deg2rad(creator.angle)))

        if isinstance(executor, FixedFogNode):
            executor_next_position_x = executor.x
            executor_next_position_y = executor.y
        else:
            executor_next_position_x = (executor.x + executor.speed * time * np.cos(np.deg2rad(executor.angle)))
            executor_next_position_y = (executor.y + executor.speed * time * np.sin(np.deg2rad(executor.angle)))
        # if executor != task.creator:
        #     print(yellow_bg(f"executor: {executor.x}, {executor.y}      creator:{creator.x}, {creator.y}        ||||||||||       {creator.speed}, {creator.angle}, {time}\n"
        #                     f"executor_next_position: {executor_next_position_x}, {executor_next_position_y}      creator_next_position:{creator_next_position_x}, {creator_next_position_y}"))
        return np.sqrt(
            (creator_next_position_y - executor_next_position_y) ** 2 +
            (creator_next_position_x - executor_next_position_x) ** 2
        )

    def can_offload_task(self, task: Task) -> bool:
        # print("oomad")
        # if task.creator.can_offload_task(task):
        #     return True

        all_fog_nodes: [str, FogLayerABC] = {**self.fixed_fog_nodes, **self.mobile_fog_nodes}
        # print(f"all_fog_nodes: {all_fog_nodes}")
        self.all_possible_nodes: dict[str, FogLayerABC] = {}
        for node in all_fog_nodes.values():
            if node.can_offload_task(task):
                self.all_possible_nodes[node.id] = node
        return len(self.all_possible_nodes) > 0

    def update(self, **kwargs: Unpack[ZoneManagerUpdate]):
        pass
