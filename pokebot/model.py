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


def load_actor_critic(path: str, n_actions: int = 26, device="cpu"):
    """Load a checkpoint, inferring input size and hidden width from the weights.
    Returns (net, obs_dim)."""
    sd = torch.load(path, map_location=device)
    hidden, obs_dim = sd["body.0.weight"].shape
    net = ActorCritic(obs_dim, n_actions, hidden=hidden).to(device)
    net.load_state_dict(sd)
    net.eval()
    return net, obs_dim


def pad_state_dict(sd: dict, obs_dim: int) -> dict:
    """Warm start from a checkpoint with fewer input features: the new input columns start at zero
    (so the network initially behaves exactly like the old one). Encoders are prefix-compatible
    (v2 is the first 811 values of v3, which are the first 967 of v4)."""
    sd = dict(sd)
    w = sd["body.0.weight"]
    if w.shape[1] > obs_dim:
        raise ValueError(f"checkpoint has {w.shape[1]} inputs, more than the requested {obs_dim}")
    if w.shape[1] < obs_dim:
        sd["body.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], obs_dim - w.shape[1], dtype=w.dtype)], 1)
    return sd
