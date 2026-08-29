import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Actor(nn.Module):
    """Deterministic DDPG policy network."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.layer_1 = nn.Linear(state_dim, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, action_dim)
        self.max_action = float(max_action)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.layer_1(state))
        x = torch.relu(self.layer_2(x))
        return self.max_action * torch.tanh(self.layer_3(x))


class Critic(nn.Module):
    """Q-network that evaluates Q(s, a)."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.layer_1 = nn.Linear(state_dim, 128)
        self.layer_2 = nn.Linear(128 + action_dim, 128)
        self.layer_3 = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state_features = torch.relu(self.layer_1(state))
        x = torch.cat([state_features, action], dim=1)
        x = torch.relu(self.layer_2(x))
        return self.layer_3(x)


class DDPGAgent:
    """
    Generic DDPG learner for the new paper-based baseline.

    This file intentionally contains no system-specific logic:
    no hard/soft-task logic, no energy, no burden model, no fog/cloud/vehicle
    rules, no action masking, no partial offloading, and no reward calculation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        actor_lr: float = 0.001,
        critic_lr: float = 0.002,
        gamma: float = 0.99,
        tau: float = 0.01,
        memory_size: int = 10000,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DDPGNew] Using device: {self.device}")

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.max_action = float(max_action)
        self.gamma = float(gamma)
        self.tau = float(tau)

        self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)

        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=float(actor_lr))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=float(critic_lr))

        self.memory = deque(maxlen=int(memory_size))

    def select_action(self, state, exploration_noise: float = 0.1) -> np.ndarray:
        """
        Return the Actor's continuous score vector.

        The ZoneManager later applies feasibility/prospective-overload masking
        and maps this vector to exactly one whole-task executor.
        """
        state = np.asarray(state, dtype=np.float32).reshape(1, -1)

        if state.shape[1] != self.state_dim:
            raise ValueError(
                f"DDPG state has {state.shape[1]} features, expected {self.state_dim}."
            )

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().reshape(-1)

        exploration_noise = float(exploration_noise)
        if exploration_noise > 0.0:
            noise = np.random.normal(
                loc=0.0,
                scale=self.max_action * exploration_noise,
                size=action.shape,
            )
            action = action + noise

        return np.clip(
            action,
            -self.max_action,
            self.max_action,
        ).astype(np.float32)

    def store_experience(self, state, action, reward, next_state, done) -> None:
        """Store one complete transition (s, a, r, s', done)."""
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        next_state = np.asarray(next_state, dtype=np.float32).reshape(-1)

        if state.size != self.state_dim:
            raise ValueError(
                f"Stored state size {state.size}, expected {self.state_dim}."
            )
        if next_state.size != self.state_dim:
            raise ValueError(
                f"Stored next_state size {next_state.size}, expected {self.state_dim}."
            )
        if action.size != self.action_dim:
            raise ValueError(
                f"Stored action size {action.size}, expected {self.action_dim}."
            )

        self.memory.append(
            (
                state.copy(),
                action.copy(),
                float(reward),
                next_state.copy(),
                bool(done),
            )
        )

    def train(self, batch_size: int = 256):
        """Run one DDPG gradient update from replay memory."""
        batch_size = int(batch_size)

        if len(self.memory) < batch_size:
            return None

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.as_tensor(
            np.asarray(states, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            np.asarray(actions, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).reshape(batch_size, self.action_dim)
        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)
        next_states = torch.as_tensor(
            np.asarray(next_states, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        # Critic update
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            target_q = rewards + self.gamma * (1.0 - dones) * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update_target_networks()

        return {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "critic_loss": float(critic_loss.detach().cpu().item()),
        }

    def _soft_update_target_networks(self) -> None:
        """Soft-update target Actor and target Critic."""
        with torch.no_grad():
            for target_param, param in zip(
                self.target_critic.parameters(),
                self.critic.parameters(),
            ):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * param.data)

            for target_param, param in zip(
                self.target_actor.parameters(),
                self.actor.parameters(),
            ):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * param.data)

    def save_model(self, filename: str = "ddpg_new_delay_baseline") -> None:
        torch.save(self.actor.state_dict(), f"{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{filename}_critic.pth")

    def load_model(self, filename: str = "ddpg_new_delay_baseline") -> None:
        self.actor.load_state_dict(
            torch.load(f"{filename}_actor.pth", map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(f"{filename}_critic.pth", map_location=self.device)
        )

        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)