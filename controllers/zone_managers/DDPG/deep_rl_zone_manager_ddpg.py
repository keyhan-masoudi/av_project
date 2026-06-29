import numpy as np
from typing import Unpack

from controllers.zone_managers.base import ZoneManagerABC, ZoneManagerUpdate
from controllers.zone_managers.DDPG.ddpg_agent import DDPGAgent
from controllers.zone_managers.deepRL.deep_rl_env import DeepRLEnvironment
from controllers.zone_managers.heuristic import HeuristicZoneManager
from models.node.fog import FogLayerABC
from models.task import Task
from config import Config
from utils.distance import get_distance


class DeepRLZoneManager_DDPG(ZoneManagerABC):
    """
    A zone manager that uses Deep Deterministic Policy Gradient (DDPG) for task offloading.
    """

    def __init__(self, zone):
        super().__init__(zone)

        self.agent = DDPGAgent(state_dim=28, action_dim=1, max_action=1.0)
        self.env = None

        try:
            self.agent.load_model("ddpg_model")
            print("[DDPG] Loaded pre-trained model.")
        except FileNotFoundError:
            print("[DDPG] No pre-trained model found, starting fresh.")

    def set_simulator(self, simulator):
        """
        Transfers the simulator reference to the environment.
        """
        self.env = DeepRLEnvironment(simulator)

    def can_offload_task(self, task: Task) -> bool:
        """
        Checks if there is an available node to offload the task.
        (This method remains unchanged)
        """
        available_nodes = list(self.fixed_fog_nodes.values()) + list(self.mobile_fog_nodes.values())
        return any(node.can_offload_task(task) for node in available_nodes)

    def assign_task(self, task: Task) -> FogLayerABC:
        pass

    def propose_candidate(self, task: Task, current_time: float):
        """
        Uses DDPG to decide where to offload a task.
        It maps the continuous output [-1, 1] to 5 discrete actions.
        """
        state = self.env._get_state(task, current_time)
        action_mask = self.env.get_action_mask(task)

        continuous_action = self.agent.select_action(state)[0]

        # Map continuous action [-1, 1] to 5 discrete bins
        if continuous_action < -0.6:
            discrete_action = 0
        elif continuous_action < -0.2:
            discrete_action = 1
        elif continuous_action < 0.2:
            discrete_action = 2
        elif continuous_action < 0.6:
            discrete_action = 3
        else:
            discrete_action = 4

        candidate_executor = None

        # Reusing the nearest fogs function from deep_rl_env
        if discrete_action == 0:
            candidate_executor = task.creator
        elif discrete_action == 1:
            candidate_executor = self.env.simulator.cloud_node
        elif discrete_action in [2, 3, 4]:
            fog_index = discrete_action - 2
            nearest_fogs = self.env._get_k_nearest_fogs(task.creator, k=3)
            if fog_index < len(nearest_fogs):
                candidate_executor = nearest_fogs[fog_index]
            else:
                candidate_executor = task.creator
        else:
            candidate_executor = task.creator

        return self, candidate_executor, continuous_action

    def update(self, **kwargs: Unpack[ZoneManagerUpdate]):
        """
        Updates the DDPG agent by training it on a batch of experiences.
        """
        self.agent.train()