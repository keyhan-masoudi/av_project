import abc
from typing import Dict, List, Tuple
from xml.etree import ElementTree
from xml.etree.ElementTree import Element

from models.base import ModelBaseABC
from models.node.fog import FixedFogNode, MobileFogNode
from models.node.user import UserNode
from models.task import Task
from models.zone import Zone
from config import Config


class SumoXMLParserABC(abc.ABC):
    def __init__(self, xml_file_path: str) -> None:
        self.xml_file_path = xml_file_path
        with open(self.xml_file_path, 'rb') as f:
            self.root: Element = ElementTree.parse(f).getroot()

    @abc.abstractmethod
    def parse(self) -> List[ModelBaseABC]:
        raise NotImplementedError


class ZoneSumoXMLParser(SumoXMLParserABC):
    def __init__(self, xml_file_path: str):
        super().__init__(xml_file_path)
        self._zones: List[Zone] = []

    def parse(self) -> List[Zone]:
        if self._zones:
            return self._zones

        zones: List[Zone] = []
        for zone_data in self.root.findall('zone'):
            zones.append(
                Zone(
                    id=zone_data.get('id'),
                    x=float(zone_data.get('x')),
                    y=float(zone_data.get('y')),
                    radius=float(zone_data.get('radius'))
                )
            )
        self._zones = zones
        return zones


class FixedFogNodeSumoXMLParser(SumoXMLParserABC):
    def __init__(self, xml_file_path: str):
        super().__init__(xml_file_path)
        self._fixed_fog_nodes: List[FixedFogNode] = []

    def parse(self) -> List[FixedFogNode]:
        if self._fixed_fog_nodes:
            return self._fixed_fog_nodes

        fixed_fog_nodes: List[FixedFogNode] = []
        for fn_data in self.root.findall('node'):
            fixed_fog_nodes.append(
                FixedFogNode(
                    id=fn_data.get('id'),
                    x=float(fn_data.get('x')),
                    y=float(fn_data.get('y')),
                    radius=float(fn_data.get('radius')),
                    power=Config.FixedFogNodeConfig.DEFAULT_COMPUTATION_POWER,
                    remaining_power=Config.FixedFogNodeConfig.DEFAULT_COMPUTATION_POWER,
                )
            )
        self._fixed_fog_nodes = fixed_fog_nodes

        return fixed_fog_nodes


class MobileNodeSumoXMLParser(SumoXMLParserABC):
    def __init__(self, chunk_path: str, chunk_number: int):
        xml_file_path = f"{chunk_path}/chunk_{chunk_number}.xml"
        super().__init__(xml_file_path)
        self._data: Dict[float, Tuple[List[UserNode], List[MobileFogNode]]] = {}  # TODO: Maybe change to int.

    def parse(self) -> Dict[float, Tuple[List[UserNode], List[MobileFogNode]]]:
        if self._data:
            return self._data

        data: Dict[float, Tuple[List[UserNode], List[MobileFogNode]]] = {}

        for time in self.root.findall('.//timestep'):
            step = float(time.get('time'))

            user_nodes: List[UserNode] = []
            mobile_fog_nodes: List[MobileFogNode] = []
            for vehicle in time.findall('vehicle'):
                parsed_data = dict(
                    id=vehicle.get('id'),
                    x=float(vehicle.get('x')),
                    y=float(vehicle.get('y')),
                    angle=float(vehicle.get('angle')),
                    speed=float(vehicle.get('speed')),
                    power=float(vehicle.get('power')),
                    remaining_power=float(vehicle.get('power')),
                    radius=float(vehicle.get('radius', Config.MobileFogNodeConfig.DEFAULT_RADIUS)),
                    frequency=float(vehicle.get('frequency')),
                    weather=1
                )
                if vehicle.get('type') == "LKW_special":  # Mobile Fog Nodes
                    parsed_data["power"] = Config.MobileFogNodeConfig.DEFAULT_COMPUTATION_POWER
                    parsed_data["remaining_power"] = Config.MobileFogNodeConfig.DEFAULT_COMPUTATION_POWER
                    parsed_data["frequency"] = Config.MobileFogNodeConfig.MOBILE_NODE_FREQUENCY
                    mobile_fog_nodes.append(MobileFogNode(**parsed_data))
                elif vehicle.get('type') == "PKW_special":  # User Nodes
                    parsed_data["power"] = Config.UserNodeConfig.DEFAULT_COMPUTATION_POWER
                    parsed_data["remaining_power"] = Config.UserNodeConfig.DEFAULT_COMPUTATION_POWER
                    parsed_data["frequency"] = Config.UserNodeConfig.USER_NODE_FREQUENCY
                    user_nodes.append(UserNode(**parsed_data))

            data[step] = (user_nodes, mobile_fog_nodes)

        self._data = data
        return data


class TaskSumoXMLParser(SumoXMLParserABC):
    def __init__(self, chunk_path: str, chunk_number: int):  # TODO: generate tasks file.
        xml_file_path = f"{chunk_path}/chunk_{chunk_number}.xml"
        super().__init__(xml_file_path)
        self._data: Dict[float, List[Task]] = {}

    def parse(self) -> Dict[float, List[Task]]:
        if self._data:
            return self._data

        data: Dict[float, List[Task]] = {}
        for time in self.root.findall('.//timestep'):
            step = float(time.get('time'))

            tasks: List[Task] = []
            for task in time.findall('task'):
                tasks.append(
                    Task(
                        id=task.get('id'),
                        release_time=step,
                        deadline=float(task.get('deadline')),
                        exec_time=float(task.get('exec_time')),
                        power=float(task.get('power')),
                        creator_id=task.get('creator'),
                        dataSize=float(task.get('dataSize')),
                        cycles_per_bit=float(task.get('cycles_per_bit'))
                    )
                )
            data[step] = tasks

        self._data = data
        return data