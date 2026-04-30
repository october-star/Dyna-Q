import random
import numpy as np

from agents.dyna_q import DynaQAgent
from agents.q_learning import QLearningAgent


class DynaQPlusAgent(DynaQAgent):
    def __init__(self, kappa=0.001, **kwargs):
        super().__init__(**kwargs)

        self.kappa = kappa
        self.global_timestep = 0
        self.last_tried = {}
        self.visited_states = []

    def _register_state(self, state):
        if state in self.visited_states:
            return

        self.visited_states.append(state)

        for action in range(self.actions):
            if (state, action) not in self.model:
                # Untried actions are modeled as zero-reward self-loops so they
                # can still be selected during planning and receive exploration bonus.
                self.model[(state, action)] = (state, 0, False)

            if (state, action) not in self.last_tried:
                self.last_tried[(state, action)] = 0

    def update(self, state, action, reward, next_state, done):
        self.global_timestep += 1

        self._register_state(state)
        if not done:
            self._register_state(next_state)

        # Real interaction uses the true reward without exploration bonus.
        QLearningAgent.update(self, state, action, reward, next_state, done)

        # Overwrite the default self-loop model entry with the observed transition.
        self.model[(state, action)] = (next_state, reward, done)
        self.last_tried[(state, action)] = self.global_timestep

        # Keep compatibility with the parent agent's bookkeeping helpers.
        if (state, action) not in self.visited_pairs:
            self.visited_pairs.append((state, action))

        for _ in range(self.planning_steps):
            if not self.visited_states:
                break

            s = random.choice(self.visited_states)
            a = random.randrange(self.actions)

            s_next, model_reward, model_done = self.model[(s, a)]
            tau = self.global_timestep - self.last_tried[(s, a)]
            bonus_reward = model_reward + self.kappa * np.sqrt(tau)

            QLearningAgent.update(self, s, a, bonus_reward, s_next, model_done)

    def clear_model(self):
        super().clear_model()
        self.last_tried.clear()
        self.visited_states.clear()
        self.global_timestep = 0
