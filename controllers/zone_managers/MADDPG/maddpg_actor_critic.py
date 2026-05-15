import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, state, mask=None):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
        return F.softmax(logits, dim=-1)

class Critic(nn.Module):
    def __init__(self, global_state_dim, global_action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(global_state_dim + global_action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, global_state, global_action):
        x = torch.cat([global_state, global_action], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)