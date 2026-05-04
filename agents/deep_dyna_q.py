import torch
import torch.nn.functional as F
from torch import nn

from agents.dqn import DQNAgent, ReplayBuffer


class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(128, 128)):
        super().__init__()

        layers = []
        input_dim = state_dim + action_dim
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, state_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state_action):
        return self.network(state_action)


class DeepDynaQAgent(DQNAgent):
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
        planning_batch_size=32,
        planning_steps=5,
        planning_start_size=1000,
        target_update_interval=200,
        replay_capacity=50000,
        hidden_dims=(128, 128),
        model_hidden_dims=(128, 128),
        model_learning_rate=1e-3,
        model_train_steps=2,
        grad_clip_norm=10.0,
        state_low=None,
        state_high=None,
        device=None,
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            learning_rate=learning_rate,
            batch_size=batch_size,
            target_update_interval=target_update_interval,
            replay_capacity=replay_capacity,
            hidden_dims=hidden_dims,
            device=device,
        )

        self.planning_batch_size = planning_batch_size
        self.planning_steps = planning_steps
        self.planning_start_size = planning_start_size
        self.model_train_steps = model_train_steps
        self.grad_clip_norm = grad_clip_norm
        self.model_network = WorldModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=model_hidden_dims,
        ).to(self.device)
        self.model_optimizer = torch.optim.Adam(
            self.model_network.parameters(), lr=model_learning_rate
        )
        self.loss_fn = nn.SmoothL1Loss()
        self.model_loss_fn = nn.SmoothL1Loss()

        self.state_low = None
        self.state_high = None
        self.state_scale = None
        if state_low is not None and state_high is not None:
            self.state_low = torch.as_tensor(state_low, dtype=torch.float32, device=self.device)
            self.state_high = torch.as_tensor(state_high, dtype=torch.float32, device=self.device)
            self.state_scale = self.state_high - self.state_low

    def normalize_state(self, states):
        if self.state_low is None or self.state_high is None:
            return states
        return 2.0 * (states - self.state_low) / self.state_scale - 1.0

    def choose_action(self, state, training=True):
        if training and torch.rand(1).item() < self.epsilon:
            return int(torch.randint(low=0, high=self.action_dim, size=(1,)).item())

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        normalized_state = self.normalize_state(state_tensor)
        with torch.no_grad():
            q_values = self.q_network(normalized_state)
        return int(torch.argmax(q_values, dim=1).item())

    def _state_action_features(self, states, actions):
        action_one_hot = F.one_hot(actions, num_classes=self.action_dim).float()
        normalized_states = self.normalize_state(states)
        return torch.cat([normalized_states, action_one_hot], dim=1)

    def _sample_batch(self, batch_size):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def _predict_next_states(self, states, actions):
        features = self._state_action_features(states, actions)
        predicted_delta = self.model_network(features)
        predicted_next_states = states + predicted_delta
        if self.state_low is not None and self.state_high is not None:
            predicted_next_states = torch.max(
                torch.min(predicted_next_states, self.state_high),
                self.state_low,
            )
        return predicted_next_states

    def _q_update_from_tensors(self, states, actions, rewards, next_states, dones):
        normalized_states = self.normalize_state(states)
        normalized_next_states = self.normalize_state(next_states)
        action_indices = actions.unsqueeze(1)
        current_q = self.q_network(normalized_states).gather(1, action_indices)
        with torch.no_grad():
            next_actions = self.q_network(normalized_next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_network(normalized_next_states).gather(1, next_actions)
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.grad_clip_norm)
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return float(loss.item())

    def _direct_rl_update(self):
        states, actions, rewards, next_states, dones = self._sample_batch(self.batch_size)
        return self._q_update_from_tensors(states, actions, rewards, next_states, dones)

    def _world_model_update(self):
        states, actions, _, next_states, _ = self._sample_batch(self.batch_size)
        target_delta = next_states - states
        predicted_next_states = self._predict_next_states(states, actions)
        predicted_delta = predicted_next_states - states
        loss = self.model_loss_fn(predicted_delta, target_delta)
        self.model_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model_network.parameters(), max_norm=self.grad_clip_norm)
        self.model_optimizer.step()
        return float(loss.item())

    def _planning_update(self, planning_reward_fn):
        planning_batch_size = min(self.planning_batch_size, len(self.replay_buffer))
        states, actions, _, _, _ = self._sample_batch(planning_batch_size)
        with torch.no_grad():
            predicted_next_states = self._predict_next_states(states, actions)
            predicted_rewards, predicted_dones = planning_reward_fn(predicted_next_states)
        return self._q_update_from_tensors(
            states,
            actions,
            predicted_rewards,
            predicted_next_states,
            predicted_dones,
        )

    def update(self, planning_reward_fn):
        min_required = max(self.batch_size, self.planning_batch_size)
        if len(self.replay_buffer) < min_required:
            return None

        direct_loss = self._direct_rl_update()
        model_losses = []
        for _ in range(self.model_train_steps):
            model_losses.append(self._world_model_update())
        model_loss = float(sum(model_losses) / len(model_losses))

        if len(self.replay_buffer) < self.planning_start_size:
            return {
                "direct_q_loss": direct_loss,
                "model_loss": model_loss,
                "planning_q_loss": 0.0,
            }

        planning_losses = []
        for _ in range(self.planning_steps):
            planning_losses.append(self._planning_update(planning_reward_fn))

        mean_planning_loss = float(sum(planning_losses) / len(planning_losses)) if planning_losses else 0.0
        return {
            "direct_q_loss": direct_loss,
            "model_loss": model_loss,
            "planning_q_loss": mean_planning_loss,
        }
