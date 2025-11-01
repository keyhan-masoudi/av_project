import os
import random
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from typing import Unpack

from controllers.zone_managers.hrl.sac import SACNetwork, SACMemory
from controllers.zone_managers.hrl.dqn import DQNNetwork, ReplayBuffer
from controllers.zone_managers.base import ZoneManagerABC, ZoneManagerUpdate
from models.node.base import MobileNodeABC
from models.node.fog import FogLayerABC
from models.task import Task
from models.zone import Zone
from utils.enums import Layer
from config import Config


class HRLZoneManager(ZoneManagerABC):
    def __init__(self, zone: Zone, dqn_lr=1e-3, sac_lr=3e-4, memory_size=10000):
        super().__init__(zone)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # DQN (Low Level)
        self.state_dim_local = 6  # Task requirements + creator resources
        self.dqn = DQNNetwork(self.state_dim_local).to(self.device)
        self.dqn_target = DQNNetwork(self.state_dim_local).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_optimizer = optim.Adam(self.dqn.parameters(), lr=dqn_lr)
        self.dqn_memory = ReplayBuffer(memory_size)
        self.dqn_target_update = 10
        self.dqn_update_counter = 0
        self.dqn_task_states = {}
        self.dqn_task_actions = {}
        self.gamma = 0.99
        self.epsilon = 0.5
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 64

        # SAC (High Level)
        self.state_dim_global = 9
        self.action_dim = 2
        self.sac = SACNetwork(self.state_dim_global, self.action_dim).to(self.device)
        self.sac_target = SACNetwork(self.state_dim_global, self.action_dim).to(self.device)
        self.sac_target.load_state_dict(self.sac.state_dict())
        self.sac_optimizer = optim.Adam(self.sac.parameters(), lr=sac_lr)
        self.sac_memory = SACMemory(memory_size)
        self.alpha = 0.2  # Temperature parameter
        self.sac_tau = 0.005  # Soft update parameter
        self.sac_task_states = {}
        self.sac_task_actions = {}


        self.all_possible_nodes = {}

    def save_checkpoint(self, path: str) -> None:
        """
        Save the current state of the DQN models and training parameters.

        Args:
            path: Directory path where to save the checkpoint
        """
        try:
            os.makedirs(path, exist_ok=True)

            # Prepare checkpoint data for both DQN and SAC
            checkpoint = {
                # DQN components
                'dqn_state_dict': self.dqn.state_dict(),
                'dqn_target_state_dict': self.dqn_target.state_dict(),
                'dqn_optimizer_state_dict': self.dqn_optimizer.state_dict(),
                'epsilon': self.epsilon,
                'dqn_hyperparameters': {
                    'gamma': self.gamma,
                    'epsilon_decay': self.epsilon_decay,
                    'epsilon_min': self.epsilon_min,
                    'batch_size': self.batch_size
                },

                # SAC components
                'sac_actor_state_dict': self.sac.state_dict(),
                'sac_target_state_dict': self.sac_target.state_dict(),
                'sac_optimizer_state_dict': self.sac_optimizer.state_dict(),
                'sac_hyperparameters': {
                    'alpha': self.alpha,
                    'sac_tau': self.sac_tau,
                    'gamma': self.gamma,
                    'batch_size': self.batch_size
                }
            }

            # Create filename
            filename = f'hrl_checkpoint_{self.zone.id}.pth'
            filepath = os.path.join(path, filename)

            # Save checkpoint
            torch.save(checkpoint, filepath)
            print(f"Successfully saved checkpoint to {filepath}")

        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
            raise

    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> None:
        """
        Load a saved checkpoint and restore the model state.

        Args:
            path: Path to the checkpoint file
            load_optimizer: Whether to load optimizer state

        Returns:
            Dict containing the loaded hyperparameters and training stats
        """
        try:
            # Load checkpoint from file
            filepath = os.path.join(path, f'hrl_checkpoint_{self.zone.id}.pth')
            checkpoint = torch.load(filepath, map_location=self.device)

            # Load DQN model states
            self.dqn.load_state_dict(checkpoint['dqn_state_dict'])
            self.dqn_target.load_state_dict(checkpoint['dqn_target_state_dict'])

            # Load SAC model states
            self.sac.load_state_dict(checkpoint['sac_actor_state_dict'])
            self.sac_target.load_state_dict(checkpoint['sac_target_state_dict'])

            # Optionally load optimizer states
            if load_optimizer:
                if 'dqn_optimizer_state_dict' in checkpoint:
                    self.dqn_optimizer.load_state_dict(checkpoint['dqn_optimizer_state_dict'])
                if 'sac_optimizer_state_dict' in checkpoint:
                    self.sac_optimizer.load_state_dict(checkpoint['sac_optimizer_state_dict'])

            # Restore DQN training state
            self.epsilon = checkpoint['epsilon']

            # Load DQN hyperparameters
            dqn_hyperparams = checkpoint['dqn_hyperparameters']
            self.gamma = dqn_hyperparams['gamma']
            self.epsilon_decay = dqn_hyperparams['epsilon_decay']
            self.epsilon_min = dqn_hyperparams['epsilon_min']
            self.batch_size = dqn_hyperparams['batch_size']

            # Load SAC hyperparameters
            sac_hyperparams = checkpoint['sac_hyperparameters']
            self.alpha = sac_hyperparams['alpha']
            self.sac_tau = sac_hyperparams['sac_tau']
            # Note: gamma and batch_size are shared between DQN and SAC
            print(f"Successfully loaded checkpoint from {filepath}")
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")

    def _get_local_state(self, task: Task) -> torch.Tensor:
        """Get state representation for local offloading decision."""
        task_power = task.power
        if task.creator.layer == Layer.USER:
            task_power *= Config.UserNodeConfig.LOCAL_OFFLOAD_POWER_OVERHEAD
        state = [
            # Task parameters
            task_power / 100.0,
            task.exec_time / 100.0,
            task.deadline / 100.0,
            # Creator parameters
            task.creator.remaining_power / 100.0,
            len(task.creator.tasks),
            task.creator.max_tasks_queue_len,
        ]
        return torch.FloatTensor(state).to(self.device)

    def _get_global_state(self, task: Task) -> torch.Tensor:
        """Get state representation for global node selection."""
        all_fog_nodes = {**self.fixed_fog_nodes, **self.mobile_fog_nodes}

        # Calculate distances to zone center
        creator_distance = np.sqrt(
            (task.creator.x - self.zone.x) ** 2 +
            (task.creator.y - self.zone.y) ** 2
        ) / self.zone.radius

        # Calculate normalized creator coordinates relative to zone center
        creator_rel_x = (task.creator.x - self.zone.x) / self.zone.radius
        creator_rel_y = (task.creator.y - self.zone.y) / self.zone.radius

        # Normalize speed (assuming some maximum speed, e.g., 30 m/s)
        MAX_SPEED = 30.0
        normalized_speed = task.creator.speed / MAX_SPEED

        # Angle is in degrees, normalize to [-1, 1]
        normalized_angle = task.creator.angle / 180.0

        # Calculate average load and std dev of loads
        loads = [len(node.tasks) for node in all_fog_nodes.values() if node.can_offload_task(task)]
        avg_load = np.mean(loads) if loads else 0
        std_load = np.std(loads) if loads else 0

        state = [
            task.power / 100.0,  # Normalized task power
            task.exec_time / 100.0,  # Normalized execution time
            creator_rel_x,  # Relative X position
            creator_rel_y,  # Relative Y position
            normalized_speed,  # Normalized speed
            normalized_angle,  # Normalized angle in degrees / 180
            creator_distance,  # Distance from zone center
            avg_load / 10.0,  # Normalized average load
            std_load / 5.0,  # Normalized load standard deviation
        ]
        return torch.FloatTensor(state).to(self.device)

    def _select_action(self, state: torch.Tensor) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 1)

        with torch.no_grad():
            q_values = self.dqn(state.unsqueeze(0))
            return q_values.argmax().item()

    def _select_node(self, task: Task, action: torch.Tensor) -> FogLayerABC | None:
        # Convert SAC action to polar coordinates within the zone
        angle_degrees = action[0].item() * 180.0  # Maps [-1,1] to [-180,180]
        angle_radians = np.deg2rad(angle_degrees)

        # Use absolute value of action[1] to ensure positive distance
        distance = (abs(action[1].item()) * self.zone.radius)

        # Convert to cartesian coordinates relative to zone center
        target_x = self.zone.x + distance * np.cos(angle_radians)
        target_y = self.zone.y + distance * np.sin(angle_radians)

        # Get current loads of all nodes
        loads = [len(node.tasks) for node in self.all_possible_nodes.values()]
        max_load = max(loads) if loads else 1
        avg_load = sum(loads) / len(loads) if loads else 0

        # Weight factors for our selection criteria
        DISTANCE_WEIGHT = 0.6
        LOAD_WEIGHT = 0.4

        best_node = None
        best_score = float('inf')

        for node in self.all_possible_nodes.values():
            # Calculate distance to target point
            predicted_x = node.x
            predicted_y = node.y
            if isinstance(node, MobileNodeABC):
                predicted_x = node.x + np.deg2rad(node.angle) * node.speed
                predicted_y = node.y + np.deg2rad(node.angle) * node.speed
            distance_to_target = np.sqrt(
                (predicted_x - target_x) ** 2 +
                (predicted_y - target_y) ** 2
            )
            distance_score = distance_to_target / self.zone.radius

            # Calculate load score
            current_load = len(node.tasks)
            if max_load:
                load_deviation = abs(current_load - avg_load) / max_load
            else:
                load_deviation = 0

            # Combined score (lower is better)
            score = DISTANCE_WEIGHT * distance_score + LOAD_WEIGHT * load_deviation

            if score < best_score:
                best_score = score
                best_node = node

        return best_node

    def _update_dqn_network(self):
        if len(self.dqn_memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.dqn_memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.float().to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.float().to(self.device)

        # Compute current Q values
        current_q = self.dqn(states).gather(1, actions.unsqueeze(1))

        # Compute target Q values
        with torch.no_grad():
            max_next_q = self.dqn_target(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # Compute loss and update
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
        self.dqn_optimizer.zero_grad()
        loss.backward()
        self.dqn_optimizer.step()

        # Update target network
        self.dqn_update_counter += 1
        if self.dqn_update_counter % self.dqn_target_update == 0:
            self.dqn_target.load_state_dict(self.dqn.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _update_sac_networks(self):
        states, actions, rewards, next_states, dones = \
            self.sac_memory.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs = self.sac.sample(next_states)
            q1_next, q2_next = self.sac_target.get_q_values(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)
            q_target = rewards + (1 - dones) * self.gamma * \
                       (q_next - self.alpha * next_log_probs)

        q1, q2 = self.sac.get_q_values(states, actions)
        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)

        # Update actor
        new_actions, log_probs = self.sac.sample(states)
        q1_new, q2_new = self.sac.get_q_values(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        # Perform updates
        self.sac_optimizer.zero_grad()
        actor_loss.backward()
        q1_loss.backward()
        q2_loss.backward()
        self.sac_optimizer.step()

        # Soft update target network
        for target_param, param in zip(self.sac_target.parameters(),
                                       self.sac.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.sac_tau) +
                param.data * self.sac_tau
            )

    def can_offload_task(self, task: Task) -> bool:
        # Update high level RL state
        all_fog_nodes = {**self.fixed_fog_nodes, **self.mobile_fog_nodes}
        self.all_possible_nodes = {
            node_id: node for node_id, node in all_fog_nodes.items()
            if node.can_offload_task(task)
        }

        # Update lower level RL state
        dqn_current_state = self._get_local_state(task)
        dqn_current_action = self._select_action(dqn_current_state)
        self.dqn_task_states[task.id] = dqn_current_state
        self.dqn_task_actions[task.id] = dqn_current_action
        if dqn_current_action == 0 and task.creator.can_offload_task(task):
            return True
        return len(self.all_possible_nodes) > 0

    def assign_task(self, task: Task) -> FogLayerABC:
        # First check if we should process locally (using existing DQN logic)
        dqn_current_action = self.dqn_task_actions.get(task.id, 1)
        if dqn_current_action == 0 and task.creator.can_offload_task(task):
            return task.creator

        # If not local, use SAC to select the best fog node
        state = self._get_global_state(task)
        with torch.no_grad():
            action, _ = self.sac.sample(state.unsqueeze(0))

        # Store state and action for later update
        self.sac_task_states[task.id] = state
        self.sac_task_actions[task.id] = action.squeeze(0)

        return self._select_node(task, action.squeeze(0))

    @staticmethod
    def _calculate_distance(creator, node) -> float:
        return np.sqrt((creator.x - node.x) ** 2 + (creator.y - node.y) ** 2)

    def update(self, **kwargs: Unpack[ZoneManagerUpdate]):
        current_task = kwargs.get('current_task')
        if not current_task:
            return
        self._update_dqn(current_task)
        self._update_sac(current_task)

    def _update_dqn(self, current_task: Task):

        # Calculate reward
        reward = self._calculate_reward(current_task)

        # Get next state
        next_state = self._get_local_state(current_task)

        # Store transition
        done = current_task.is_completed or current_task.is_deadline_missed
        dqn_current_state = self.dqn_task_states.pop(current_task.id, None)
        dqn_current_action = self.dqn_task_actions.pop(current_task.id, None)
        self.dqn_memory.push(
            dqn_current_state,
            dqn_current_action,
            reward,
            next_state,
            done
        )

        # Update network
        self._update_dqn_network()

    def _update_sac(self, task: Task):
        if task.id not in self.sac_task_states:
            return

        # Calculate reward
        reward = self._calculate_sac_reward(task)

        # Get next state
        next_state = self._get_global_state(task)

        # Store transition
        done = task.is_completed or task.is_deadline_missed
        state = self.sac_task_states.pop(task.id)
        action = self.sac_task_actions.pop(task.id)

        self.sac_memory.push(state, action, reward, next_state, done)

        # Update networks if we have enough samples
        if len(self.sac_memory) > self.batch_size:
            self._update_sac_networks()

    def _calculate_sac_reward(self, task: Task) -> float:
        reward = 0.0

        if task.is_completed:
            reward += 10.0
            # Add bonus for completing well before deadline
            time_margin = task.deadline - task.finish_time
            reward += max(0.0, time_margin / task.deadline) * 5.0

            # Add bonus for good load balancing
            executor_load = len(task.executor.tasks)
            all_loads = [len(node.tasks) for node in self.all_possible_nodes.values()]
            avg_load = np.mean(all_loads) if all_loads else 0
            load_diff = abs(executor_load - avg_load)
            reward += 5.0 * (1.0 - min(1.0, load_diff / 5.0))

        if task.is_deadline_missed:
            reward -= 20.0

        if task.has_migrated:
            reward -= 5.0

        return reward

    @staticmethod
    def _calculate_reward(task: Task) -> float:
        reward = 0.0

        # Small negative reward per time step to encourage faster completion
        reward -= 0.1

        # Check task status
        if task.executor == task.creator:
            # Ongoing local processing
            reward += 5  # Small positive reward to consider local processing

        # Completed or failed conditions
        if task.has_migrated:
            reward -= 5.0  # Penalty for task migration

            if task.executor.layer == Layer.CLOUD:
                reward -= 15.0  # Penalty for offloading to cloud

        completion_time = task.finish_time
        deadline_margin = task.deadline - completion_time
        reward += 10.0 + max(0.0, deadline_margin / task.deadline) * 5.0  # Reward for meeting deadline

        if task.is_deadline_missed:  # Penalty for missing deadline
            reward -= 50.0

        return reward
