import torch
from agents.deep_dyna_q import DeepDynaQAgent
from agents.ensemble_world_model import EnsembleWorldModel

class EnsembleDeepDynaQAgent(DeepDynaQAgent):
    def __init__(self, K=3, lambda_penalty=1.0, **kwargs):
        kwargs.pop('warmup_steps', None) # Fixes TypeError[cite: 9]
        super().__init__(**kwargs)
        self.K = K
        self.lambda_penalty = lambda_penalty
        self.last_planning_disagreement = 0.0 # Internal state for tracking
        
        self.ensemble = EnsembleWorldModel(
            K=K, 
            state_dim=kwargs['state_dim'], 
            action_dim=kwargs['action_dim'],
            hidden_dims=kwargs.get('model_hidden_dims', (128, 128)),
            device=self.device
        )
        lr = kwargs.get('model_learning_rate', 1e-3)
        self.model_optimizers = [torch.optim.Adam(m.parameters(), lr=lr) for m in self.ensemble.models]

    def _world_model_update(self):
        """Updates K models independently[cite: 1]."""
        total_loss = 0
        for i in range(self.K):
            states, actions, _, next_states, _ = self._sample_batch(self.batch_size)
            target_delta = next_states - states
            features = self._state_action_features(states, actions)
            predicted_delta = self.ensemble.models[i](features)
            loss = self.model_loss_fn(predicted_delta, target_delta)
            self.model_optimizers[i].zero_grad(); loss.backward(); self.model_optimizers[i].step()
            total_loss += loss.item()
        return total_loss / self.K

    def _planning_update(self, planning_reward_fn):
        """Returns ONLY the float loss to stay compatible with parent class[cite: 9]."""
        states, actions, _, _, _ = self._sample_batch(self.planning_batch_size)
        with torch.no_grad():
            u = self.ensemble.get_uncertainty(states, actions, self.normalize_state)
            self.last_planning_disagreement = u.mean().item() # Store for the update() call
            features = self._state_action_features(states, actions)
            next_states = states + self.ensemble.models[0](features)
            r_base, dones = planning_reward_fn(next_states)
            r_penalized = r_base - self.lambda_penalty * u # Uncertainty penalty[cite: 1]
        return self._q_update_from_tensors(states, actions, r_penalized, next_states, dones)

    def update(self, planning_reward_fn):
        """Injects disagreement data into the results dictionary[cite: 1]."""
        loss_info = super().update(planning_reward_fn)
        if isinstance(loss_info, dict):
            loss_info["avg_disagreement"] = self.last_planning_disagreement
        return loss_info