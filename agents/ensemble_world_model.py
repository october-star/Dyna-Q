import torch
from torch import nn
import torch.nn.functional as F
from agents.deep_dyna_q import WorldModel

class EnsembleWorldModel(nn.Module):
    def __init__(self, K, state_dim, action_dim, hidden_dims=(128, 128), device="cpu"):
        super().__init__()
        self.K = K
        self.action_dim = action_dim # Fixes dimension mismatch[cite: 1, 9]
        self.models = nn.ModuleList([
            WorldModel(state_dim, action_dim, hidden_dims).to(device) 
            for _ in range(K)
        ])
        self.device = device

    def get_uncertainty(self, states, actions, normalize_fn):
        """Calculates max pairwise disagreement u(s,a)."""
        action_one_hot = F.one_hot(actions, num_classes=self.action_dim).float()
        features = torch.cat([normalize_fn(states), action_one_hot], dim=1)
        
        with torch.no_grad():
            preds = torch.stack([m(features) for m in self.models])
            max_disagreement = torch.zeros(preds.shape[1], device=self.device)
            for i in range(self.K):
                for j in range(i + 1, self.K):
                    dist = torch.norm(preds[i] - preds[j], dim=1)**2
                    max_disagreement = torch.max(max_disagreement, dist)
        return max_disagreement.unsqueeze(1)