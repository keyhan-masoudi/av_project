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
        self.agent = SACAgent(state_dim=28, action_dim=5)

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
        Returns a tuple containing the zone manager and the selected executor node.
        """
        state = self.env._get_state(task, current_time)

        if hasattr(self.env, "get_action_mask"):
            action_mask = self.env.get_action_mask(task)
        else:
            action_mask = None

        action = self.agent.select_action(state, mask=action_mask)
        candidate_executor = None

        # Reusing the action mapping logic identical to DeepRLZoneManager
        if action == 0:
            candidate_executor = task.creator
        elif action == 1:
            candidate_executor = self.env.simulator.cloud_node
        elif action in [2, 3, 4]:
            fog_index = action - 2
            # Reuse the nearest fogs function from deep_rl_env
            nearest_fogs = self.env._get_k_nearest_fogs(task.creator, k=3)
            if fog_index < len(nearest_fogs):
                candidate_executor = nearest_fogs[fog_index]
            else:
                candidate_executor = task.creator
        else:
            candidate_executor = task.creator

        return self, candidate_executor

    def update(self, **kwargs: Unpack[ZoneManagerUpdate]):
        """
        Updates the RL agent by training it with experiences from the replay buffer.
        """
        self.agent.train()

    def assign_task(self, task: Task) -> FogLayerABC:
        """
                Deprecated or less relevant in the current simulation flow
                as the simulator makes the final assignment decision.
                """
        pass