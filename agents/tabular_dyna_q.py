import random
from collections import defaultdict

import numpy as np


class TabularMountainCarDynaQAgent:
    def __init__(
        self,
        actions,
        discretizer,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1,
        planning_steps=10,
    ):
        self.actions = actions
        self.discretizer = discretizer
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps

        self.q_table = defaultdict(lambda: np.zeros(self.actions))
        self.model = {}
        self.visited_states = []
        self.state_actions = {}

    def discretize_state(self, state):
        return self.discretizer.discretize(state)

    def choose_action(self, state, training=True):
        discrete_state = self.discretize_state(state)
        return self.choose_action_discrete(discrete_state, training=training)
    
    def decay_epsilon(self, min_epsilon=0.01, decay_rate=0.995):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    def choose_action_discrete(self, discrete_state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.actions)

        q_values = self.q_table[discrete_state]
        max_q = np.max(q_values)
        max_actions = [action for action, value in enumerate(q_values) if value == max_q]
        return int(np.random.choice(max_actions))

    def update(self, state, action, reward, next_state, done):
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        self.update_discrete(discrete_state, action, reward, discrete_next_state, done)

    def update_discrete(self, discrete_state, action, reward, discrete_next_state, done):
        self._q_update(discrete_state, action, reward, discrete_next_state, done)
        self.model[(discrete_state, action)] = (reward, discrete_next_state, done)
        self._register_transition(discrete_state, action)
        self._planning_update()

    def _q_update(self, discrete_state, action, reward, discrete_next_state, done):
        current_q = self.q_table[discrete_state][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[discrete_next_state])

        self.q_table[discrete_state][action] = current_q + self.alpha * (target - current_q)

    def _register_transition(self, discrete_state, action):
        if discrete_state not in self.visited_states:
            self.visited_states.append(discrete_state)

        if discrete_state not in self.state_actions:
            self.state_actions[discrete_state] = []

        if action not in self.state_actions[discrete_state]:
            self.state_actions[discrete_state].append(action)

    def _planning_update(self):
        if not self.model:
            return

        model_keys = list(self.model.keys())

        for _ in range(self.planning_steps):
            sampled_state, sampled_action = random.choice(model_keys)
            reward, next_state, done = self.model[(sampled_state, sampled_action)]
            self._q_update(sampled_state, sampled_action, reward, next_state, done)
