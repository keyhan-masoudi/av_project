import torch
import torch.nn as nn
from torch.distributions import Categorical

class MAPPOActor(nn.Module):
    def __init__(self, local_state_dim, action_dim):
        super(MAPPOActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(local_state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.action_head = nn.Linear(128, action_dim)

    def forward(self, local_state, action_mask=None):
        x = self.net(local_state)
        logits = self.action_head(x)
        
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)
            
        return Categorical(logits=logits)

class MAPPOCritic(nn.Module):
    def __init__(self, global_state_dim):
        super(MAPPOCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, global_state):
        return self.net(global_state)