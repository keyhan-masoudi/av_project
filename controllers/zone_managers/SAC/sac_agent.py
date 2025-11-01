import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


# Helper function for soft target updates
def soft_update(target_model, local_model, tau):
    """
    Softly update model parameters:
    θ_target = τ*θ_local + (1 - τ)*θ_target
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class Actor(nn.Module):
    """
    Actor (Policy) Network: Maps state to action probabilities.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Use softmax to get a probability distribution over discrete actions
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs


class Critic(nn.Module):
    """
    Critic (Q-Value) Network: Maps state to Q-values for each action.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class SACAgent:
    """
    Soft Actor-Critic Agent for Discrete Action Spaces.
    """

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, target_entropy=None):
        self.gamma = gamma  # Discount factor
        self.tau = tau  # Soft update coefficient
        self.alpha = alpha  # Initial entropy temperature

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor Network
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critic Networks (using two critics to reduce overestimation)
        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic1_target = Critic(state_dim, action_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)

        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.critic2_target = Critic(state_dim, action_dim).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        # Automatic Entropy Tuning
        if target_entropy is None:
            # Heuristic for discrete actions
            self.target_entropy = -np.log(1.0 / action_dim) * 0.98
        else:
            self.target_entropy = target_entropy

        self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # Experience Replay Memory
        self.memory = deque(maxlen=10000)

    def select_action(self, state):
        """Select an action based on the current policy."""
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
        return action.item()

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=256):
        """Train the Actor, Critic, and Alpha."""
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # --- Update Critic Networks ---
        with torch.no_grad():
            next_action_probs = self.actor(next_states)
            next_log_action_probs = torch.log(next_action_probs + 1e-8)

            q1_target_next = self.critic1_target(next_states)
            q2_target_next = self.critic2_target(next_states)
            min_q_target_next = torch.min(q1_target_next, q2_target_next)

            # Calculate the target Q-value using entropy
            target_v = (next_action_probs * (min_q_target_next - self.alpha * next_log_action_probs)).sum(dim=1,
                                                                                                          keepdim=True)
            target_q = rewards + self.gamma * (1 - dones) * target_v

        # Q-value predictions from current critics
        q1_pred = self.critic1(states).gather(1, actions)
        q2_pred = self.critic2(states).gather(1, actions)

        # Critic loss
        critic1_loss = F.mse_loss(q1_pred, target_q)
        critic2_loss = F.mse_loss(q2_pred, target_q)

        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # --- Update Actor Network and Alpha ---
        action_probs = self.actor(states)
        log_action_probs = torch.log(action_probs + 1e-8)

        with torch.no_grad():
            q1_values = self.critic1(states)
            q2_values = self.critic2(states)
            min_q_values = torch.min(q1_values, q2_values)

        # Actor loss
        actor_loss = (action_probs * (self.alpha * log_action_probs - min_q_values)).sum(dim=1).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha (entropy temperature) loss
        # Use detach() on log_action_probs as we don't want to backprop through the actor here
        alpha_loss = -(self.log_alpha * (log_action_probs + self.target_entropy).detach()).mean()

        # Update alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # --- Soft Update Target Networks ---
        soft_update(self.critic1_target, self.critic1, self.tau)
        soft_update(self.critic2_target, self.critic2, self.tau)

    def save_model(self, path="sac_model"):
        """Save trained models."""
        torch.save(self.actor.state_dict(), f"{path}_actor.pth")
        torch.save(self.critic1.state_dict(), f"{path}_critic1.pth")
        torch.save(self.critic2.state_dict(), f"{path}_critic2.pth")

    def load_model(self, path="sac_model"):
        """Load trained models."""
        self.actor.load_state_dict(torch.load(f"{path}_actor.pth", map_location=self.device))
        self.critic1.load_state_dict(torch.load(f"{path}_critic1.pth", map_location=self.device))
        self.critic2.load_state_dict(torch.load(f"{path}_critic2.pth", map_location=self.device))
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())