from collections import defaultdict
import gc
from typing import Type

from NoiseConfigs.utilsFunctions import UtilsFunc
from controllers.zone_managers.base import ZoneManagerABC
from controllers.zone_managers.heuristic import HeuristicZoneManager
# from controllers.zone_managers.heuristic2 import HeuristicZoneManager2
from controllers.zone_managers.random import RandomZoneManager
# from controllers.zone_managers.hrl import HRLZoneManager
from controllers.zone_managers.only_cloud import OnlyCloudZoneManager
from controllers.zone_managers.only_fog import OnlyFogZoneManager
from controllers.zone_managers.only_local import OnlyLocalZoneManager
from controllers.zone_managers.deepRL.deep_rl_zone_manager import DeepRLZoneManager
from controllers.zone_managers.MADDPG.deep_rl_zone_manager_maddpg import DeepRLZoneManagerMADDGP
from controllers.zone_managers.MAPPO.deep_rl_zone_manager_mappo import DeepRLZoneManagerMAPPO
from controllers.zone_managers.DDPG.deep_rl_zone_manager_ddpg import DeepRLZoneManager_DDPG
from controllers.zone_managers.PPO.deep_rl_zone_manager_PPO import DeepRLZoneManagerPPO
from controllers.zone_managers.SAC.deep_rl_zone_manager_sac import DeepRLZoneManagerSAC
from controllers.zone_managers.greedy import GreedyZoneManager
from utils.xml_parser import *


class Loader:
    ALGORITHM_MAP: Dict[str, Type[ZoneManagerABC]] = {
        Config.ZoneManagerConfig.ALGORITHM_RANDOM: RandomZoneManager,
        Config.ZoneManagerConfig.ALGORITHM_HEURISTIC: HeuristicZoneManager,
        Config.ZoneManagerConfig.ALGORITHM_ONLY_CLOUD: OnlyCloudZoneManager,
        Config.ZoneManagerConfig.ALGORITHM_ONLY_FOG: OnlyFogZoneManager,
        Config.ZoneManagerConfig.ALGORITHM_ONLY_LOCAL: OnlyLocalZoneManager,
        Config.ZoneManagerConfig.ALGORITHM_DEEP_RL: DeepRLZoneManager,
        Config.ZoneManagerConfig.ALGORITHM_MAPPO: DeepRLZoneManagerMAPPO,
        Config.ZoneManagerConfig.ALGORITHM_MADDPG: DeepRLZoneManagerMADDGP,
        Config.ZoneManagerConfig.ALGORITHM_DDPG: DeepRLZoneManager_DDPG,
        Config.ZoneManagerConfig.ALGORITHM_PPO: DeepRLZoneManagerPPO,
        Config.ZoneManagerConfig.ALGORITHM_SAC: DeepRLZoneManagerSAC,
        Config.ZoneManagerConfig.ALGORITHM_GREEDY: GreedyZoneManager
    }

    def __init__(
            self,
            zone_file: str,
            fixed_fn_file: str,
            mobile_file: str,
            task_file: str,
            checkpoint_path: str,
            hard_task_file: str = "./data/hard_tasks",
    ):
        self.chunk_size = Config.CHUNK_SIZE
        
        start_time = Config.SimulatorConfig.SIMULATION_START_TIME
        self.current_chunk = int(start_time) // self.chunk_size
        
        self.zone_parser = ZoneSumoXMLParser(zone_file)
        self.fixed_fn_parser = FixedFogNodeSumoXMLParser(fixed_fn_file)
        
        self.mobile_chunk_path = mobile_file
        self.task_chunk_path = task_file
        self.hard_task_chunk_path = hard_task_file
        self.checkpoint_path = checkpoint_path

        self.mobile_node_parser = MobileNodeSumoXMLParser(mobile_file, self.current_chunk)
        self.task_parser = TaskSumoXMLParser(task_file, self.current_chunk)
        self.hard_task_parser = TaskSumoXMLParser(hard_task_file, self.current_chunk) if hard_task_file else None

    def __load_next_chunk(self, time_step: float):
        target_chunk = self.get_chunk(time_step)
        
        if target_chunk != self.current_chunk:
            if hasattr(self, 'mobile_node_parser') and self.mobile_node_parser:
                self.mobile_node_parser._data.clear()
            if hasattr(self, 'task_parser') and self.task_parser:
                self.task_parser._data.clear()
            if hasattr(self, 'hard_task_parser') and self.hard_task_parser:
                self.hard_task_parser._data.clear()
                
            gc.collect()

            self.mobile_node_parser = MobileNodeSumoXMLParser(self.mobile_chunk_path, target_chunk)
            self.task_parser = TaskSumoXMLParser(self.task_chunk_path, target_chunk)
            if self.hard_task_chunk_path:
                self.hard_task_parser = TaskSumoXMLParser(self.hard_task_chunk_path, target_chunk)
                
            self.current_chunk = target_chunk
    def get_chunk(self, time_step: float) -> int:
        return round(time_step) // self.chunk_size

    def load_zones(self) -> Dict[str, ZoneManagerABC]:
        zone_managers: Dict[str, ZoneManagerABC] = {}

        zones = self.zone_parser.parse()

        # note: removed!
        # partitions = UtilsFunc.load_partitions("generated_hex_partitions")
        # factory_partitions = self.findAllFactoryPartitions(partitions)

        for zone in zones:
            # print(f"zone : {zone}")
            # todo: add HRL
            # zonePartition = UtilsFunc.find_partition(partitions, zone.x, zone.y)
            # print(f"zonePartition: {zonePartition.centerX, zonePartition.centerY}")
            # for fp in factory_partitions:
            #     print(f"factory_partitions: {fp.centerX, fp.centerY }")
            #
            # if any(zonePartition.centerX == fp.centerX and zonePartition.centerY == fp.centerY for fp in factory_partitions):
            #     print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOMMMMMMMMMMMMMMMMMMMAAAAAAAAAAAAAAAAAADDDDDDDDDDDDDDDDDDD")
            #     zone_manager_cls = self.ALGORITHM_MAP[Config.ZoneManagerConfig.ALGORITHM_RANDOM]

            # else:
            # print("yes")
            zone_manager_cls = self.ALGORITHM_MAP[Config.ZoneManagerConfig.DEFAULT_ALGORITHM]

            zone_manager_obj = zone_manager_cls(zone)

            # If using DeepRL, set the simulator reference
            # if isinstance(zone_manager_obj, DeepRLZoneManager):
            #     zone_manager_obj.env.simulator = self

            zone_managers[zone.id] = zone_manager_obj

            # if isinstance(zone_manager_obj, HRLZoneManager):
            #     zone_manager_obj.load_checkpoint(self.checkpoint_path)

        return zone_managers

    def load_fixed_zones(self) -> Dict[str, FixedFogNode]:
        fixed_fog_nodes: Dict[str, FixedFogNode] = {}

        for fixed_node in self.fixed_fn_parser.parse():
            fixed_fog_nodes[fixed_node.id] = fixed_node
        return fixed_fog_nodes

    def load_mobile_fog_nodes(self, time_step: float) -> Dict[str, MobileFogNode]:
        self.__load_next_chunk(time_step)
        mobile_fog_nodes: Dict[str, MobileFogNode] = {}

        if time_step < Config.SimulatorConfig.SIMULATION_DURATION - 1:
            for mobile_node in self.mobile_node_parser.parse()[time_step][1]:
                mobile_fog_nodes[mobile_node.id] = mobile_node

        return mobile_fog_nodes

    def load_user_nodes(self, time_step: float) -> Dict[str, UserNode]:
        self.__load_next_chunk(time_step)
        user_fog_nodes: Dict[str, UserNode] = {}

        if time_step < Config.SimulatorConfig.SIMULATION_DURATION:
            for user_node in self.mobile_node_parser.parse()[time_step][0]:
                user_fog_nodes[user_node.id] = user_node
        return user_fog_nodes

    def load_nodes_tasks(self, time_step: float) -> Dict[str, List[Task]]:
        """Load soft (aperiodic) tasks for the given simulation step."""
        self.__load_next_chunk(time_step)
        tasks: Dict[str, List[Task]] = defaultdict(list)

        for task in self.task_parser.parse().get(time_step, []):
            tasks[task.creator_id].append(task)
        return tasks

    def load_nodes_hard_tasks(self, time_step: float) -> Dict[str, List[Task]]:
        """Load pre-generated periodic hard tasks for the given simulation step."""
        if self.hard_task_parser is None:
            return {}

        self.__load_next_chunk(time_step)
        tasks: Dict[str, List[Task]] = defaultdict(list)

        for task in self.hard_task_parser.parse().get(time_step, []):
            task.is_hard = True
            tasks[task.creator_id].append(task)
        return tasks

    # def findAllFactoryPartitions(self, partitions):
    #     temp = []
    #     for partition in partitions:
    #         if partition.is_factory:
    #             temp.append(partition)
    #     return temp
