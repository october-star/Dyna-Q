# import random
# import numpy as np
# from agents.dyna_q import DynaQAgent
# from agents.q_learning import QLearningAgent


# class DynaQPlusAgent(DynaQAgent):

#     def __init__(self, kappa=0.001, **kwargs):
#         # Initialize parent Dyna-Q agent
#         super().__init__(**kwargs)

#         # Exploration bonus weight (κ)
#         self.kappa = kappa

#         # Global timestep counter: incremented once per real environment step
#         self.global_timestep = 0

#         # Tracks the last real-world timestep each (state, action) was tried
#         # Initialised to 0 so unseen pairs accumulate bonus from the start
#         self.last_tried = {}

#         # Initialize the list to track unique visited states
#         self.visited_states = []

#     def update(self, state, action, reward, next_state, done):
#         # Advance global clock
#         self.global_timestep += 1

#         # Initialise last_tried for any new (state, action) pair
#         if (state, action) not in self.last_tried:
#             self.last_tried[(state, action)] = 0

#         # Direct RL update using real experience — no bonus applied here
#         # Call grandparent directly to avoid triggering DynaQAgent's planning loop
#         QLearningAgent.update(self, state, action, reward, next_state, done)

#         # Store the observed transition in the environment model
#         self.model[(state, action)] = (next_state, reward)
        

#         # # Record unique (state, action) pairs for planning sampling
#         if state not in self.visited_states:
#             self.visited_states.append(state)

#         # Record that this pair was just tried in the real world
#         self.last_tried[(state, action)] = self.global_timestep


#         # Perform n planning steps using the learned model with exploration bonus
#         for _ in range(self.planning_steps):
#             if not self.visited_pairs:
#                 break

#             # Randomly select a past (state, action) pair
#             #s, a = random.choice(self.visited_pairs)
#             s = random.choice(self.visited_states)
#             a = random.randint(0, self.num_actions - 1)
        

#             # # Retrieve the stored transition from the model
#             # r, s_next = self.model[(s, a)]

#             # Retrieve the stored transition from the model, OR default if unseen
#             if (s, a) in self.model:
#                 r, s_next = self.model[(s, a)]
#             else:
#                 # If we've never tried this action, assume it does nothing and gives 0 reward
#                 r, s_next = 0, s

#             # Compute τ: steps since (s, a) was last tried in the real world
#             tau = self.global_timestep - self.last_tried.get((s, a), 0)

#             # Augmented reward: r' = r + κ * sqrt(τ)
#             r_plus = r + self.kappa * np.sqrt(tau)

#             # Update Q-value using the augmented reward
#             QLearningAgent.update(self, s, a, r_plus, s_next, done=False)


import random
import numpy as np
from agents.dyna_q import DynaQAgent
from agents.q_learning import QLearningAgent


class DynaQPlusAgent(DynaQAgent):
    def __init__(self, kappa=0.001, **kwargs):
        # Initialize parent Dyna-Q agent
        super().__init__(**kwargs)

        # Exploration bonus coefficient
        self.kappa = kappa

        # Global timestep: increments once per real environment step
        self.global_timestep = 0

        # last_tried[(state, action)] = last real timestep when (s,a) was executed
        self.last_tried = {}

        # Track visited states for Dyna-Q+ planning
        # Dyna-Q+ samples a visited state, then considers all actions
        self.visited_states = []

    def update(self, state, action, reward, next_state, done):
        # Advance real-world clock
        self.global_timestep += 1

        # First time visiting this state: register it and initialize all actions
        if state not in self.visited_states:
            self.visited_states.append(state)

            for a in range(self.actions):
                # For unseen actions, assume self-loop with zero reward
                if (state, a) not in self.model:
                    self.model[(state, a)] = (state, 0, False)  # (next_state, reward, done)

                # Untried actions accumulate bonus from timestep 0
                if (state, a) not in self.last_tried:
                    self.last_tried[(state, a)] = 0

        # Real experience update (NO exploration bonus here)
        QLearningAgent.update(self, state, action, reward, next_state, done)

        # Store the real transition in the model
        self.model[(state, action)] = (next_state, reward, done)

        # Mark when this (state, action) was last tried in the real environment
        self.last_tried[(state, action)] = self.global_timestep

        # Planning with exploration bonus
        for _ in range(self.planning_steps):
            if not self.visited_states:
                break

            # Sample a previously visited state
            s = random.choice(self.visited_states)

            # Sample any action, including never-tried ones
            a = random.randint(0, self.actions - 1)

            # Retrieve model transition
            s_next, r, model_done = self.model[(s, a)]

            # Time since this action was last tried for real
            tau = self.global_timestep - self.last_tried[(s, a)]

            # Add Dyna-Q+ exploration bonus
            r_plus = r + self.kappa * np.sqrt(tau)

            # Planning update uses augmented reward
            QLearningAgent.update(self, s, a, r_plus, s_next, model_done)

    def clear_model(self):
        super().clear_model()
        self.last_tried.clear()
        self.visited_states.clear()
        self.global_timestep = 0