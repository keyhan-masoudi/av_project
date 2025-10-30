import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np

torch.autograd.set_detect_anomaly(True)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.actor_head = nn.Linear(128, action_dim)

        self.critic_head = nn.Linear(128, 1)

    def forward(self, state):
        x = self.shared_layers(state)

        action_probs = torch.softmax(self.actor_head(x), dim=-1)

        state_value = self.critic_head(x)

        return action_probs, state_value


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, K_epochs=4, clip_epsilon=0.2,
                 gae_lambda=0.95):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.memory = []

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam([
            {'params': self.policy.shared_layers.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor_head.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic_head.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.criterion = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            action_probs, state_val = self.policy_old(state_tensor)

            dist = Categorical(action_probs)
            action = dist.sample()

            action_log_prob = dist.log_prob(action)

        return action.item(), action_log_prob.cpu().numpy().flatten(), state_val.cpu().numpy().flatten()

    def store_experience(self, state, action, reward, done, log_prob, value):
        self.memory.append((state, action, reward, done, log_prob, value))

    def update(self):
        rewards = []
        discounted_reward = 0

        states, actions, old_log_probs, old_values, dones = [], [], [], [], []
        for state, action, reward, done, log_prob, value in self.memory:
            states.append(state)
            actions.append(action)
            old_log_probs.append(log_prob)
            old_values.append(value)
            dones.append(done)
            rewards.append(reward)

        returns = []
        advantages = []
        last_gae_lam = 0
        for i in reversed(range(len(rewards))):
            next_non_terminal = 1 - dones[i]
            next_value = old_values[i + 1] if i + 1 < len(old_values) else 0

            delta = rewards[i] + self.gamma * next_value * next_non_terminal - old_values[i]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages.insert(0, last_gae_lam)
            returns.insert(0, last_gae_lam + old_values[i])

        old_states = torch.FloatTensor(np.array(states)).to(self.device)
        old_actions = torch.LongTensor(np.array(actions)).to(self.device).view(-1, 1)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device).view(-1, 1)
        advantages = torch.FloatTensor(advantages).to(self.device).view(-1, 1)
        returns = torch.FloatTensor(returns).to(self.device).view(-1, 1)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        for _ in range(self.K_epochs):
            action_probs, state_values = self.policy(old_states)
            dist = Categorical(action_probs)

            new_log_probs = dist.log_prob(old_actions.squeeze())
            dist_entropy = dist.entropy()

            ratios = torch.exp(new_log_probs.view(-1, 1) - old_log_probs.view(-1, 1))

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2)

            critic_loss = self.criterion(state_values, returns)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy.view(-1, 1)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.memory.clear()

    def save_model(self, filename="ppo_model.pth"):
        torch.save(self.policy.state_dict(), filename)

    def load_model(self, filename="ppo_model.pth"):
        self.policy.load_state_dict(torch.load(filename, map_location=self.device))
        self.policy_old.load_state_dict(self.policy.state_dict())