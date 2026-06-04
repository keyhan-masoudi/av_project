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

    # note : not important function
    def assign_task(self, task: Task) -> FogLayerABC:
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        pass

    def propose_candidate(self, task: Task, current_time: float):
        """
        Uses the Deep RL agent to decide where to offload a task.
        It just suggests a node and return (ZN, node)
        """
        state = self.env._get_state(task, current_time)

        if hasattr(self.env, "get_action_mask"):
            action_mask = self.env.get_action_mask(task)
        else:
            action_mask = None

        action = self.agent.select_action(state, mask=action_mask)
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
                candidate_executor = task.creator
        else:
            candidate_executor = task.creator

        return self, candidate_executor

    def update(self, **kwargs: Unpack[ZoneManagerUpdate]):
        """
        Updates the RL agent after each simulation step.
        """
        self.agent.train()  # Train the agent periodically
        if np.random.random() < 0.05:  # Update target network occasionally
            self.agent.update_target_network()
