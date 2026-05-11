from agents.dyna_q import DynaQAgent


class TabularMountainCarDynaQAgent:
    def __init__(self, actions, discretizer, **kwargs):
        self.discretizer = discretizer
        self.agent = DynaQAgent(actions=actions, **kwargs)

    def discretize_state(self, state):
        return self.discretizer.discretize(state)

    def choose_action(self, state, training=True):
        discrete_state = self.discretize_state(state)
        return self.agent.choose_action(discrete_state, training=training)

    def update(self, state, action, reward, next_state, done):
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)

        return self.agent.update(
            discrete_state,
            action,
            reward,
            discrete_next_state,
            done
        )

    @property
    def epsilon(self):
        return self.agent.epsilon

    @epsilon.setter
    def epsilon(self, value):
        self.agent.epsilon = value

    def decay_epsilon(self, min_epsilon=0.01, decay_rate=0.995):
        self.agent.epsilon = max(min_epsilon, self.agent.epsilon * decay_rate)

    @property
    def alpha(self):
        return self.agent.alpha

    @alpha.setter
    def alpha(self, value):
        self.agent.alpha = value

    @property
    def gamma(self):
        return self.agent.gamma

    @gamma.setter
    def gamma(self, value):
        self.agent.gamma = value

    def get_avg_td_error(self):
        return self.agent.get_avg_td_error()

    def reset_statistics(self):
        self.agent.reset_statistics()

    def get_model_size(self):
        return self.agent.get_model_size()

    def get_visited_pairs_count(self):
        return self.agent.get_visited_pairs_count()

    def clear_model(self):
        self.agent.clear_model()

    @property
    def q_table(self):
        return self.agent.q_table