from typing_extensions import Unpack

from controllers.zone_managers.base import ZoneManagerABC, ZoneManagerUpdate
from controllers.zone_managers.PPO.ppo_agent import PPOAgent
from controllers.zone_managers.deepRL.deep_rl_env import DeepRLEnvironment
from models.node.fog import FogLayerABC
from models.task import Task
from config import Config
from utils.distance import get_distance


class DeepRLZoneManagerPPO(ZoneManagerABC):
    def __init__(self, zone):
        super().__init__(zone)

        self.agent = PPOAgent(state_dim=5, action_dim=3)
        self.env = None

        try:
            self.agent.load_model()
            print("[PPO] Loaded pre-trained model.")
        except:
            print("[PPO] No pre-trained model found, starting fresh.")

    def set_simulator(self, simulator):
        self.env = DeepRLEnvironment(simulator)

    def propose_candidate(self, task: Task, current_time: float):
        state = self.env._get_state(task)
        action, log_prob, value = self.agent.select_action(state)

        if action == 0:
            candidate_executor = task.creator
        elif action == 1:
            candidate_executor = self._get_best_fog_node(task)
        else:
            candidate_executor = self.env.simulator.cloud_node

        return self, candidate_executor, (state, action, log_prob, value)

    def update(self, **kwargs: Unpack[ZoneManagerUpdate]):

        pass

    def can_offload_task(self, task: Task) -> bool:
        """
        Checks if there is an available node to offload the task.
        """
        available_nodes = list(self.fixed_fog_nodes.values()) + list(self.mobile_fog_nodes.values())
        return any(node.can_offload_task(task) for node in available_nodes)

    def assign_task(self, task: Task) -> FogLayerABC:
        # won't use in code, i write this because of abstract function
        return self._get_best_fog_node(task)

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