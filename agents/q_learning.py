import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(self,
                 actions=4,  # Number of discrete actions (Up, Down, Left, Right)
                 alpha=0.1,  # Learning rate (0 < α ≤ 1)
                 gamma=0.95,  # Discount factor (0 ≤ γ ≤ 1)
                 epsilon=0.1):  # Exploration rate (probability of random action)

        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-table: default dict returns array of zeros for unseen states
        # Key: state tuple (row, col)
        # Value: numpy array of Q-values for each action [Q(up), Q(down), Q(left), Q(right)]
        self.q_table = defaultdict(lambda: np.zeros(actions))

        # Track statistics for debugging
        self.update_count = 0
        self.total_td_error = 0.0

    def get_q(self, state, action):
        return self.q_table[state][action]

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
        current_q = self.get_q(state, action)

        if done:
            # Terminal state: no future rewards
            # TD_target = immediate reward only
            target = reward
        else:
            # Non-terminal: r + γ * max_{a'} Q(s', a')
            # This is the Bellman Optimality Equation
            max_next_q = np.max(self.q_table[next_state])
            target = reward + self.gamma * max_next_q

        # Calculate temporal difference error
        td_error = target - current_q

        # Update Q-value
        new_q = current_q + self.alpha * td_error
        self.q_table[state][action] = new_q

        # Update statistics
        self.update_count += 1
        self.total_td_error += abs(td_error)

        return td_error

    def get_avg_td_error(self):
        if self.update_count == 0:
            return 0.0
        return self.total_td_error / self.update_count

    def reset_statistics(self):
        """Reset tracking statistics (useful for new environment phase)."""
        self.update_count = 0
        self.total_td_error = 0.0

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_alpha(self, alpha):
        self.alpha = alpha