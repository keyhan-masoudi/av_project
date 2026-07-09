from typing import Unpack

from controllers.zone_managers.base import ZoneManagerABC, ZoneManagerUpdate
from controllers.zone_managers.PPO.ppo_agent import PPOAgent
from controllers.zone_managers.deepRL.deep_rl_env import DeepRLEnvironment
from models.node.fog import FogLayerABC
from models.task import Task


class DeepRLZoneManagerPPO(ZoneManagerABC):
    def __init__(self, zone):
        super().__init__(zone)

        self.agent = PPOAgent(state_dim=28, action_dim=5)
        self.env = None

        try:
            self.agent.load_model()
            print("[PPO] Loaded pre-trained model.")
        except:
            print("[PPO] No pre-trained model found, starting fresh.")

    def set_simulator(self, simulator):
        self.env = DeepRLEnvironment(simulator)

    def propose_candidate(self, task: Task, current_time: float):
        state = self.env._get_state(task, current_time)
        action_mask = self.env.get_action_mask(task)

        action, log_prob, value = self.agent.select_action(state, action_mask=action_mask)

        # Save state and action inside the task for proper reward calculation later
        task.rl_state = state
        task.rl_action = action

        candidate_executor = None

        if action == 0:
            candidate_executor = task.creator
        elif action == 1:
            candidate_executor = self.env.simulator.cloud_node
        elif action in [2, 3, 4]:
            fog_index = action - 2
            nearest_fogs = self.env._get_k_nearest_fogs(task.creator, k=3)

            if fog_index < len(nearest_fogs):
                candidate_executor = nearest_fogs[fog_index]
            else:
                candidate_executor = task.creator  # Fallback to local if fog does not exist

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
        pass
