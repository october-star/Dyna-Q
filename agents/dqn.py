import random
from collections import deque

import numpy as np
import torch
from torch import nn


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                float(done),
            )
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(128, 128)):
        super().__init__()

        layers = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        learning_rate=1e-3,
        batch_size=64,
        target_update_interval=200,
        replay_capacity=50000,
        hidden_dims=(128, 128),
        device=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.q_network = QNetwork(state_dim, action_dim, hidden_dims=hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dims=hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(replay_capacity)
        self.training_steps = 0

    def choose_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return int(np.random.randint(self.action_dim))

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1, keepdim=True).values
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
