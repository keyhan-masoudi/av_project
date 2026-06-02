import numpy as np
from typing import Unpack

from controllers.zone_managers.base import ZoneManagerABC, ZoneManagerUpdate
from controllers.zone_managers.deepRL.deep_rl_agent import DeepRLAgent
from controllers.zone_managers.deepRL.deep_rl_env import DeepRLEnvironment
from controllers.zone_managers.heuristic import HeuristicZoneManager
from models.node.fog import FogLayerABC
from models.task import Task
from config import Config
from utils.distance import get_distance


class DeepRLZoneManager(ZoneManagerABC):
    """
    A zone manager that uses Deep Reinforcement Learning for task offloading.
    """

    def __init__(self, zone):
        super().__init__(zone)

        # Initialize Deep RL Environment and Agent
        self.agent = DeepRLAgent(state_dim=28, action_dim=5)

        self.env = None

        # Load pre-trained model if available
        try:
            self.agent.load_model()
            print("[DeepRL] Loaded pre-trained model.")
        except:
            print("[DeepRL] No pre-trained model found, starting fresh.")

    def set_simulator(self, simulator):
        """
        Transfers the simulator reference to the environment and the agent.
        """
        self.env = DeepRLEnvironment(simulator)

    def can_offload_task(self, task: Task) -> bool:
        """
        Checks if there is an available node to offload the task.
        """
        available_nodes = list(self.fixed_fog_nodes.values()) + list(self.mobile_fog_nodes.values())
        # if self.env.simulator and self.env.simulator.cloud_node:
        #     available_nodes.append(self.env.simulator.cloud_node)
        return any(node.can_offload_task(task) for node in
                   available_nodes)
        # NOTE :I removed "or self.env.simulator.cloud_node.can_offload_task(task)"

    # note : not important function
    def assign_task(self, task: Task) -> FogLayerABC:
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        state = self.env._get_state(task)
        action = self.agent.select_action(state)

        if action == 0:
            candidate_executor = task.creator
        elif action == 1:
            candidate_executor = self._get_best_fog_node(task)
        else:
            candidate_executor = self.env.simulator.cloud_node

        return candidate_executor

    # todo: should remove usage of _get_best_fog_node
    def propose_candidate(self, task: Task, current_time: float):
        """
        Uses Deep RL to decide where to offload a task.
        It just suggests a node and return (ZN, node)
        """
        state = self.env._get_state(task, current_time)
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
        return self, candidate_executor

    # def _get_best_fog_node(self, task):
    #     """
    #     Finds the best fog node to offload the task based on available resources.
    #     """
    #     # print("nooooo")
    #     fog_nodes = list(self.fixed_fog_nodes.values()) + list(self.mobile_fog_nodes.values())
    #     fog_nodes = [node for node in fog_nodes if node.can_offload_task(task)]
    #     # note: that was min , i changed another one too
    #     return max(fog_nodes, key=lambda n: n.remaining_power, default=None)

    def _get_best_fog_node(self, task):
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
        Updates the RL agent after each simulation step.
        """
        self.agent.train()  # Train the agent periodically
        if np.random.random() < 0.05:  # Update target network occasionally
            self.agent.update_target_network()
