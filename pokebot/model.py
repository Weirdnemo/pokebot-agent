from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

NEG_INF = -1e9


class ActorCritic(nn.Module):
    """MLP actor-critic with illegal-action masking.

    v0 is feed-forward. Because the opponent's sets are hidden, the next step up
    is a GRU/LSTM (or transformer over the turn history) in front of the heads.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 512):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        h = self.body(obs)
        logits = self.pi(h).masked_fill(mask < 0.5, NEG_INF)
        return Categorical(logits=logits), self.v(h).squeeze(-1)
