import random
import numpy as np
from collections import defaultdict

class TabularEnsembleDynaAgent:
    def __init__(self, actions=4, K=3, alpha=0.1, gamma=0.95, epsilon=0.1, planning_steps=10):
        """
        Discrete Tabular Ensemble Agent based on whiteboard specifications.
        MS = Set of K independent tabular transition models
        QS = Set of K distinct action-value Q-tables
        """
        self.actions = actions
        self.K = K
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps
        
        # Whiteboard: MS = {M1, M2, M3}
        self.models = [{} for _ in range(K)]
        
        # Whiteboard: QS = {Q1, Q2, Q3}
        self.q_tables = [defaultdict(lambda: np.zeros(actions)) for _ in range(K)]
        
        # Track uniquely visited states and state-actions independently per ensemble member
        self.visited_states = [[] for _ in range(K)]
        self.state_actions = [defaultdict(list) for _ in range(K)]

    def choose_action(self, state, training=True):
        """
        Consensus action selection: Computes max action across the sum of all Q-tables.
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.actions)
        
        # Aggregate valuations across the discrete Q-table ensemble
        combined_q = np.zeros(self.actions)
        for k in range(self.K):
            combined_q += self.q_tables[k][state]
            
        max_q = np.max(combined_q)
        max_actions = [a for a, q in enumerate(combined_q) if q == max_q]
        return int(np.random.choice(max_actions))

    def update(self, state, action, reward, next_state, done):
        """
        Updates models and applies independent planning updates to a randomly selected table.
        """
        # 1. Direct Real-World Update: Apply ground-truth transition to all K Q-tables
        for k in range(self.K):
            current_q = self.q_tables[k][state][action]
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(self.q_tables[k][next_state])
                
            self.q_tables[k][state][action] += self.alpha * (target - current_q)
            
            # Record direct transition in model k
            self.models[k][(state, action)] = (next_state, reward, done)
            self._register_transition(k, state, action)

        # 2. Ensemble Planning Phase (Whiteboard Loop)
        for _ in range(self.planning_steps):
            # Whiteboard: n <- random {1, 2, 3} (re-indexed to 0, 1, 2 for Python)
            n = random.randint(0, self.K - 1)
            
            if not self.models[n]:
                continue
                
            # Whiteboard: Sample state uniformly from states visited by model n
            s = random.choice(self.visited_states[n])
            a = random.choice(self.state_actions[n][s])
            
            # Whiteboard: simulate(pi, M) -> get outcome from model n
            s_next, r, model_done = self.models[n][(s, a)]
            
            # Whiteboard: update(Q) -> Update ONLY the selected Q-table
            current_q = self.q_tables[n][s][a]
            if model_done:
                target = r
            else:
                target = r + self.gamma * np.max(self.q_tables[n][s_next])
                
            self.q_tables[n][s][a] += self.alpha * (target - current_q)

    def _register_transition(self, k, state, action):
        """Internal helper to manage historical samples per ensemble index."""
        if state not in self.visited_states[k]:
            self.visited_states[k].append(state)
        if action not in self.state_actions[k][state]:
            self.state_actions[k][state].append(action)

    def get_ensemble_disagreement(self):
        """
        Calculates tabular disagreement as the average variance of Q-valuations 
        across all visited states.
        """
        all_visited = set()
        for k in range(self.K):
            all_visited.update(self.visited_states[k])
            
        if not all_visited:
            return 0.0
            
        total_variance = 0.0
        for state in all_visited:
            state_qs = []
            for k in range(self.K):
                state_qs.append(self.q_tables[k][state])
            # Variance across the ensemble dimension per action, averaged across actions
            total_variance += np.mean(np.var(state_qs, axis=0))
            
        return total_variance / len(all_visited)