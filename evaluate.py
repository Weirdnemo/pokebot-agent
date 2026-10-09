"""Evaluate a checkpoint (greedy policy) against the baseline bots.

    python evaluate.py --ckpt checkpoints/latest.pt --n 100
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from pokebot.env import make_env
from pokebot.model import load_actor_critic

OPPONENTS = {
    "random": RandomPlayer,
    "maxpower": MaxBasePowerPlayer,
    "heuristic": SimpleHeuristicsPlayer,
}


def play(net, opp_cls, n, fmt, device, obs_dim):
    env = make_env(opp_cls(start_listening=False), battle_format=fmt, obs_dim=obs_dim)
    wins = 0
    for _ in range(n):
        obs, _ = env.reset()
        done = False
        while not done:
            o = torch.as_tensor(obs["observation"], dtype=torch.float32, device=device).unsqueeze(0)
            m = torch.as_tensor(obs["action_mask"], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                dist, _ = net(o, m)
            a = dist.probs.argmax(-1).item()
            obs, r, term, trunc, _ = env.step(np.int64(a))
            done = term or trunc
        wins += int(bool(env.env.battle1.won))
    env.close()
    return wins / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--format", default="gen9randombattle")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, obs_dim = load_actor_critic(args.ckpt, 26 if "gen9" in args.format else 10, device)
    for name, cls in OPPONENTS.items():
        print(f"vs {name:10s} win rate: {play(net, cls, args.n, args.format, device, obs_dim):.2f}", flush=True)


if __name__ == "__main__":
    main()
