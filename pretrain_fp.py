"""Distil Foul Play's search policy into our network (data from tools/collect_foulplay.py).

    python pretrain_fp.py --data data/fp_500.npz --out checkpoints_fp
    python pretrain_fp.py --data data/fp_500.npz data/fp_more.npz --init checkpoints/bc.pt

Policy loss: cross-entropy against Foul Play's whole search distribution (soft targets).
`--hard` instead uses only the move it finally picked (ablation: does the soft target help?).
Value loss: MSE to the battle outcome (+1 win / -1 loss for Foul Play's side).
The train/validation split is by BATTLE, never by decision, because decisions inside one battle
are strongly correlated and a decision-level split would leak.
Rows with a single legal action are dropped from the policy loss (nothing to learn).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pokebot.encoder import OBS_DIM
from pokebot.model import ActorCritic


def load(paths):
    parts, off = [], 0
    for p in paths:
        d = np.load(p)
        b = d["battle"].astype(np.int64) + off
        off = b.max() + 1
        parts.append(dict(obs=d["obs"], mask=d["mask"], act=d["act"].astype(np.int64),
                          policy=d["policy"], n_legal=d["n_legal"], battle=b,
                          outcome=d["outcome"].astype(np.float32), meta=str(d["meta"])))
        print(f"{p}: {len(d['obs'])} decisions, {len(np.unique(b))} battles, meta={parts[-1]['meta']}")
    return {k: np.concatenate([q[k] for q in parts]) for k in parts[0] if k != "meta"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--init", default=None, help="start from this checkpoint (e.g. checkpoints/bc.pt)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=4, help="stop after this many epochs without validation improvement")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--hard", action="store_true", help="hard-label ablation")
    ap.add_argument("--vcoef", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints_fp")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    D = load(args.data)
    n = len(D["obs"])
    battles = np.unique(D["battle"])
    rng.shuffle(battles)
    n_val_b = max(1, int(len(battles) * args.val_frac))
    val_b = set(battles[:n_val_b].tolist())
    is_val = np.array([b in val_b for b in D["battle"]])
    multi = D["n_legal"] > 1
    tr = np.where(~is_val)[0]
    va = np.where(is_val)[0]
    print(f"{n} decisions, {len(battles)} battles; train {len(tr)} / val {len(va)} "
          f"(val battles {n_val_b}); {(~multi).sum()} single-action rows kept for the value loss only")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
    X, M, P, A, V, MU = (T(D["obs"]), T(D["mask"]), T(D["policy"]), T(D["act"], torch.int64),
                         T(D["outcome"]), T(multi.astype(np.float32)))
    net = ActorCritic(OBS_DIM, M.shape[1]).to(device)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=device))
        print(f"initialised from {args.init}")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    Path(args.out).mkdir(exist_ok=True)

    def policy_loss(idx):
        dist, v = net(X[idx], M[idx])
        logp = dist.logits.log_softmax(-1)
        if args.hard:
            ce = -logp.gather(1, A[idx][:, None]).squeeze(1)
        else:
            ce = -(P[idx] * logp.clamp(min=-30)).sum(-1)
        pl = (ce * MU[idx]).sum() / MU[idx].sum().clamp(min=1)
        vl = F.mse_loss(v, V[idx])
        return pl, vl, dist

    def val_metrics():
        idx = T(va, torch.int64)
        with torch.no_grad():
            pl, vl, dist = policy_loss(idx)
            m = MU[idx] > 0
            top1_search = (dist.probs.argmax(-1) == P[idx].argmax(-1))[m].float().mean().item()
            top1_pick = (dist.probs.argmax(-1) == A[idx])[m].float().mean().item()
            # entropy of the teacher and student, to see whether we are over/under-confident
            ent_t = -(P[idx] * P[idx].clamp(min=1e-9).log()).sum(-1)[m].mean().item()
            ent_s = dist.entropy()[m].mean().item()
        return pl.item(), vl.item(), top1_search, top1_pick, ent_t, ent_s

    # Reference points for reading the numbers below.
    m_va = multi[va]
    ceiling = (D["policy"][va].argmax(-1) == D["act"][va])[m_va].mean() if m_va.any() else float("nan")
    chance = (1.0 / D["n_legal"][va][m_va]).mean() if m_va.any() else float("nan")
    print(f"reference: random-legal agreement ~{chance:.3f}; a perfect copy of the argmax would agree "
          f"with Foul Play's sampled pick {ceiling:.3f} of the time")
    if len(va):
        pl, vl, t1s, t1p, et, es = val_metrics()
        print(f"epoch  0 (before training) | val policy_ce={pl:.3f} value_mse={vl:.3f} "
              f"agree_argmax={t1s:.3f} agree_pick={t1p:.3f} H_teacher={et:.2f} H_student={es:.2f}", flush=True)

    best, bad = 1e9, 0
    for ep in range(args.epochs):
        rng.shuffle(tr)
        tot = 0.0
        for s in range(0, len(tr), args.batch):
            mb = T(tr[s:s + args.batch], torch.int64)
            pl, vl, _ = policy_loss(mb)
            loss = pl + args.vcoef * vl
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(mb)
        if len(va):
            pl, vl, t1s, t1p, et, es = val_metrics()
            print(f"epoch {ep+1:2d}/{args.epochs} train={tot/len(tr):.3f} | val policy_ce={pl:.3f} "
                  f"value_mse={vl:.3f} agree_argmax={t1s:.3f} agree_pick={t1p:.3f} "
                  f"H_teacher={et:.2f} H_student={es:.2f}", flush=True)
            if pl < best:
                best, bad = pl, 0
                torch.save(net.state_dict(), f"{args.out}/fp_best.pt")
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"early stop: no validation improvement for {args.patience} epochs")
                    break
        torch.save(net.state_dict(), f"{args.out}/fp.pt")
    print(f"saved {args.out}/fp.pt (last epoch) and fp_best.pt (lowest validation policy loss)")


if __name__ == "__main__":
    main()
