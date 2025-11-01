import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 32):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)  # 2 actions: local or offload

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (torch.stack(states), torch.tensor(actions),
                torch.tensor(rewards), torch.stack(next_states),
                torch.tensor(dones))

    def __len__(self):
        return len(self.buffer)
