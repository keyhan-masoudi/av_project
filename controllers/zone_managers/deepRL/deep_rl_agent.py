import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class DQN(nn.Module):
    """
    Neural network for Deep Q-Learning.
    """
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DeepRLAgent:
    """
    Deep Q-Learning Agent for Task Offloading.
    """
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration probability
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.lr = lr

        # Experience Replay Memory
        self.memory = deque(maxlen=10000)

        print(torch.__version__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.is_available())
        print(torch.backends.cudnn.enabled)
        print(torch.backends.cudnn.version())
        print(self.device)

        # Neural Networks
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        # Optimizer and Loss Function
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def select_action(self, state, mask=None):
        """
        Select an action using ε-greedy strategy.
        """
        if random.random() < self.epsilon:
            if mask is not None:
                mask_np = np.array(mask)
                valid_actions = np.where(mask_np == 1.0)[0]
                if len(valid_actions) > 0:
                    return np.random.choice(valid_actions).item()
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            if mask is not None:
                mask_tensor = torch.FloatTensor(mask).to(self.device)
                q_values = q_values.masked_fill(mask_tensor == 0, -1e9)
        return torch.argmax(q_values).item()  # Exploit

    def store_experience(self, state, action, reward, next_state, done):
        """
        Store experience in memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=256):
        """
        Train the Deep Q-Network.
        """
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute Q-values
        q_values = self.model(states).gather(1, actions).squeeze()

        # Compute Target Q-values
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Update Model
        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print(f"------------------------------- epsilon: {self.epsilon}")

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        """
        Update target model parameters.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, filename="deep_rl_model.pth"):
        """
        Save trained model.
        """
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename="deep_rl_model.pth"):
        """
        Load trained model.
        """
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
        self.target_model.load_state_dict(self.model.state_dict())
