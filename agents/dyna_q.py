import random
from agents.q_learning import QLearningAgent
import numpy as np

class DynaQAgent(QLearningAgent):

    def __init__(self, planning_steps=5, **kwargs):
        # Initialize parent Q-Learning agent
        super().__init__(**kwargs)

        # Dyna-Q specific parameters
        self.planning_steps = planning_steps  # n: number of planning steps

        # Environment model: stores observed transitions
        # Format: model[(state, action)] = (next_state, reward)
        self.model = {}

        # Track all visited (state, action) pairs for random sampling during planning
        # This ensures all past experiences have a chance to be replayed
        self.visited_pairs = []

    def choose_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            # EXPLORATION: Try a random action to discover better paths
            return np.random.randint(self.actions)
        else:
            # EXPLOITATION: Choose the action with highest Q-value
            q_values = self.q_table[state]

            # Handle ties by randomly selecting among max-valued actions
            # This prevents bias toward the first maximum action
            max_q = np.max(q_values)
            max_actions = [a for a, q in enumerate(q_values) if q == max_q]

            return np.random.choice(max_actions)

    def update(self, state, action, reward, next_state, done):
        # Update Q-value using the real experience (same as standard Q-Learning)
        super().update(state, action, reward, next_state, done)

        # Store the observed transition in the environment model
        # This builds a deterministic model of the environment
        self.model[(state, action)] = (next_state, reward)

        # Record this (state, action) pair for future planning
        # Only store unique pairs to avoid duplicates in sampling
        if (state, action) not in self.visited_pairs:
            self.visited_pairs.append((state, action))

        # Perform n planning steps using the learned model
        # Each step randomly replays a past experience and updates Q-value
        for _ in range(self.planning_steps):
            # If no experiences have been recorded yet, skip planning
            if not self.visited_pairs:
                break

            # Randomly select a past (state, action) pair
            # Uniform sampling ensures all experiences are revisited equally
            s, a = random.choice(self.visited_pairs)

            # Retrieve the imagined outcome from the model
            s_next, r = self.model[(s, a)]

            # Update Q-value using the imagined experience
            # Note: done=False because we don't know if s_next is terminal
            # In practice, assuming non-terminal works well for planning
            super().update(s, a, r, s_next, done=False)

    def get_model_size(self):
        return len(self.model)

    def get_visited_pairs_count(self):
        return len(self.visited_pairs)

    def clear_model(self):
        self.model.clear()
        self.visited_pairs.clear()

