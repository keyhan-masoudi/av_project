import torch
import torch.optim as optim
import numpy as np
from controllers.zone_managers.MAPPO.mappo_actor_critic import MAPPOActor, MAPPOCritic

class MAPPOController:
    def __init__(self, num_agents, local_state_dim=28, action_dim=5, lr_actor=3e-4, lr_critic=1e-3, clip_epsilon=0.2, update_epochs=4):
        self.num_agents = num_agents
        self.local_state_dim = local_state_dim
        self.global_state_dim = num_agents * local_state_dim
        self.action_dim = action_dim
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = MAPPOActor(self.local_state_dim, action_dim).to(self.device)
        self.critic = MAPPOCritic(self.global_state_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.memory = []

    def select_actions(self, local_states, global_state, masks=None):
        local_states_tensor = torch.FloatTensor(np.array(local_states)).to(self.device)
        global_state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
        masks_tensor = torch.FloatTensor(np.array(masks)).to(self.device) if masks else None

        with torch.no_grad():
            dist = self.actor(local_states_tensor, action_mask=masks_tensor)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

            value = self.critic(global_state_tensor).squeeze()

        return actions.cpu().numpy(), log_probs.cpu().numpy(), value.cpu().numpy().item()

    def store_experience(self, local_state, global_state, action, log_prob, value, reward):
        self.memory.append((local_state, global_state, action, log_prob, value, reward))

    def train(self, batch_size=128):
        if len(self.memory) < batch_size:
            return

        local_states, global_states, actions, old_log_probs, values, rewards = zip(*self.memory)

        local_states = torch.FloatTensor(np.array(local_states)).to(self.device)
        global_states = torch.FloatTensor(np.array(global_states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device)
        values = torch.FloatTensor(np.array(values)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)

        advantages = rewards - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.update_epochs):
            dist = self.actor(local_states)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()

            state_values = self.critic(global_states).squeeze()
            critic_loss = torch.nn.MSELoss()(state_values, rewards)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        self.memory.clear()