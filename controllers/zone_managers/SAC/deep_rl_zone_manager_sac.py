import numpy as np
from typing import Unpack

from controllers.zone_managers.base import ZoneManagerABC, ZoneManagerUpdate
from controllers.zone_managers.SAC.sac_agent import SACAgent
from controllers.zone_managers.deepRL.deep_rl_env import DeepRLEnvironment
from models.node.fog import FogLayerABC
from models.task import Task
from config import Config
from utils.distance import get_distance


class DeepRLZoneManagerSAC(ZoneManagerABC):
    """
    A zone manager that uses Deep Reinforcement Learning (SAC) for task offloading.
    """

    def __init__(self, zone):
        super().__init__(zone)

        self.env = None
        self.agent = SACAgent(state_dim=5, action_dim=3)

        try:
            self.agent.load_model(path="sac_model")
            print("[DeepRL-SAC] Loaded pre-trained model.")
        except FileNotFoundError:
            print("[DeepRL-SAC] No pre-trained model found, starting fresh.")

    def set_simulator(self, simulator):
        """
        Transfers the simulator reference to the environment.
        """
        self.env = DeepRLEnvironment(simulator)

    def can_offload_task(self, task: Task) -> bool:
        """
        Checks if there is an available node to offload the task.
        """
        available_nodes = list(self.fixed_fog_nodes.values()) + list(self.mobile_fog_nodes.values())
        return any(node.can_offload_task(task) for node in available_nodes)

    def propose_candidate(self, task: Task, current_time: float):
        """
        Uses Deep RL (SAC) to decide where to offload a task.
        It suggests a node and returns (self, node).
        """
        state = self.env._get_state(task)
        if hasattr(self.env, "get_action_mask"):
            action_mask = self.env.get_action_mask(task)
        else:
            action_mask = None


        action = self.agent.select_action(state, mask=action_mask)

        candidate_executor = None
        if action == 0:
            candidate_executor = task.creator
        elif action == 1:
            candidate_executor = self._get_best_fog_node(task)
        else:
            candidate_executor = self.env.simulator.cloud_node

        return self, candidate_executor

    def _get_best_fog_node(self, task):
        """
        Finds the best fog node to offload the task based on a scoring mechanism
        that considers both computational power and distance.
        """
        creator = task.creator
        all_fog_nodes = list(self.fixed_fog_nodes.values()) + list(self.mobile_fog_nodes.values())
        eligible_nodes = [node for node in all_fog_nodes if node.can_offload_task(task)]

        if not eligible_nodes:
            return None

        if len(eligible_nodes) == 1:
            return eligible_nodes[0]

        distances_by_id = {
            node.id: get_distance(node.x, node.y, creator.x, creator.y)
            for node in eligible_nodes
        }

        min_dist = min(distances_by_id.values())
        max_dist = max(distances_by_id.values())

        mobile_node_ids = {node.id for node in self.mobile_fog_nodes.values()}

        def calculate_score(node):
            if node.id in mobile_node_ids:
                normalized_power = node.power / Config.MobileFogNodeConfig.DEFAULT_COMPUTATION_POWER
            else:
                normalized_power = node.power / Config.FixedFogNodeConfig.DEFAULT_COMPUTATION_POWER

            distance = distances_by_id[node.id]

            if max_dist == min_dist:
                normalized_distance = 0.0
            else:
                normalized_distance = (distance - min_dist) / (max_dist - min_dist)
            distance_score = 1.0 - normalized_distance

            final_score = (0.5 * normalized_power) + (0.5 * distance_score)
            return final_score

        chosen_node = max(
            eligible_nodes,
            key=calculate_score,
            default=None
        )

        return chosen_node

    def update(self, **kwargs: Unpack[ZoneManagerUpdate]):
        """
        Updates the RL agent by training it with experiences from the replay buffer.
        """
        self.agent.train()

    def assign_task(self, task: Task) -> FogLayerABC:
        """
        Note: This method is less relevant in the current simulation flow
        as the simulator makes the final assignment decision, but it demonstrates
        the full logic of the zone manager.
        """
        state = self.env._get_state(task)
        if hasattr(self.env, "get_action_mask"):
            action_mask = self.env.get_action_mask(task)
        else:
            action_mask = None

        action = self.agent.select_action(state, mask=action_mask)

        if action == 0:
            candidate_executor = task.creator
        elif action == 1:
            candidate_executor = self._get_best_fog_node(task)
        else:
            candidate_executor = self.env.simulator.cloud_node

        return candidate_executor