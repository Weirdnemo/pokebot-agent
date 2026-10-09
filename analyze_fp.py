"""Where does our network disagree with Foul Play?  (validation battles only)

    python analyze_fp.py --ckpt checkpoints_fp_big/fp_best.pt \
        --data data/fp_500.npz data/fp_a.npz data/fp_b.npz data/fp_c.npz data/fp_d.npz data/fp_e.npz data/fp_f.npz

Uses the same by-battle validation split as pretrain_fp.py (same --seed / --val-frac), so run it
with the same --data list and order as the training run. Reports agreement with Foul Play's top
search choice by action type, by how confident the search was, by turn, and by number of legal
actions. Purpose: tell apart "needs better features / hidden-information beliefs" from "needs
more data or a bigger model".
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from pokebot.model import load_actor_critic
from pretrain_fp import load


def kind(a):  # our 26-action space: 0-5 switch, 6-9 move, 22-25 move + terastallize
    return np.where(a < 6, 0, np.where(a < 22, 1, 2))


KIND = ["switch", "move", "tera move"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    D = load(args.data)
    rng = np.random.default_rng(args.seed)
    battles = np.unique(D["battle"])
    rng.shuffle(battles)
    val_b = set(battles[: max(1, int(len(battles) * args.val_frac))].tolist())
    sel = np.array([b in val_b for b in D["battle"]]) & (D["n_legal"] > 1)
    X, M, P, N, T = D["obs"][sel], D["mask"][sel], D["policy"][sel], D["n_legal"][sel], D["turn"][sel]

    net, obs_dim = load_actor_critic(args.ckpt, M.shape[1])
    assert obs_dim == X.shape[1], f"checkpoint expects {obs_dim} inputs but the data has {X.shape[1]}"
    with torch.no_grad():
        dist, _ = net(torch.as_tensor(X), torch.as_tensor(M, dtype=torch.float32))
    pred = dist.probs.argmax(-1).numpy()
    tgt = P.argmax(-1)
    conf = P.max(-1)
    ok = pred == tgt
    print(f"\n{sel.sum()} validation decisions with >1 legal action; overall agreement {ok.mean():.3f}\n")

    def row(name, m):
        if m.sum():
            print(f"  {name:<22} n={m.sum():5d} ({m.mean() * 100:4.1f}% of decisions)  agreement {ok[m].mean():.3f}")

    print("By Foul Play's top choice type:")
    for k in range(3):
        row(KIND[k], kind(tgt) == k)
    print("\nConfusion (rows = Foul Play's type, columns = our type), % of each row:")
    for k in range(3):
        m = kind(tgt) == k
        if m.sum():
            print(f"  {KIND[k]:<10}", "  ".join(f"{KIND[j]} {100 * (kind(pred)[m] == j).mean():4.1f}%" for j in range(3)))
    print("\nBy how peaked Foul Play's search policy was (its top option's share):")
    for lo, hi in [(0, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
        row(f"top share {lo:.1f}-{min(hi, 1):.1f}", (conf >= lo) & (conf < hi))
    print("\nBy turn:")
    for lo, hi in [(0, 4), (4, 11), (11, 21), (21, 999)]:
        row(f"turn {lo + 1}-{hi if hi < 999 else '+'}", (T >= lo) & (T < hi))
    print("\nBy number of legal actions:")
    for lo, hi in [(2, 3), (3, 5), (5, 8), (8, 99)]:
        row(f"{lo}-{hi - 1 if hi < 99 else '+'} legal", (N >= lo) & (N < hi))


if __name__ == "__main__":
    main()
