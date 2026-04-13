import random
import numpy as np
from agents.dyna_q import DynaQAgent
from agents.q_learning import QLearningAgent


class DynaQPlusAgent(DynaQAgent):

    def __init__(self, kappa=0.001, **kwargs):
        # Initialize parent Dyna-Q agent
        super().__init__(**kwargs)

        # Exploration bonus weight (κ)
        self.kappa = kappa

        # Global timestep counter: incremented once per real environment step
        self.global_timestep = 0

        # Tracks the last real-world timestep each (state, action) was tried
        # Initialised to 0 so unseen pairs accumulate bonus from the start
        self.last_tried = {}

    def update(self, state, action, reward, next_state, done):
        # Advance global clock
        self.global_timestep += 1

        # Initialise last_tried for any new (state, action) pair
        if (state, action) not in self.last_tried:
            self.last_tried[(state, action)] = 0

        # Direct RL update using real experience — no bonus applied here
        # Call grandparent directly to avoid triggering DynaQAgent's planning loop
        QLearningAgent.update(self, state, action, reward, next_state, done)

        # Store the observed transition in the environment model
        self.model[(state, action)] = (next_state, reward)

        # Record unique (state, action) pairs for planning sampling
        if (state, action) not in self.visited_pairs:
            self.visited_pairs.append((state, action))

        # Record that this pair was just tried in the real world
        self.last_tried[(state, action)] = self.global_timestep

        # Perform n planning steps using the learned model with exploration bonus
        for _ in range(self.planning_steps):
            if not self.visited_pairs:
                break

            # Randomly select a past (state, action) pair
            s, a = random.choice(self.visited_pairs)

            # Retrieve the stored transition from the model
            s_next, r = self.model[(s, a)]

            # Compute τ: steps since (s, a) was last tried in the real world
            tau = self.global_timestep - self.last_tried.get((s, a), 0)

            # Augmented reward: r' = r + κ * sqrt(τ)
            r_plus = r + self.kappa * np.sqrt(tau)

            # Update Q-value using the augmented reward
            QLearningAgent.update(self, s, a, r_plus, s_next, done=False)