"""Behavior-cloning warm start from a scripted teacher (default: SimpleHeuristicsPlayer).

    python pretrain_bc.py --battles 3000                 # -> checkpoints/bc.pt
    python train_ppo.py --opponent mixed --resume checkpoints/bc.pt --lr 1e-4 --out checkpoints_bc

The teacher plays as our agent through the normal env, so every (observation,
legal-action mask, chosen action) triple uses exactly the encoder the policy
sees. Discounted reward-to-go is stored too, so the value head starts calibrated
instead of random (that keeps early PPO updates from wrecking the cloned policy).
The teacher's own win rate per opponent is printed: that is the bar to beat.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from poke_env.environment import SinglesEnv
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from pokebot.encoder import OBS_DIM
from pokebot.env import make_env
from pokebot.model import ActorCritic

PLAYERS = {"random": RandomPlayer, "maxpower": MaxBasePowerPlayer, "heuristic": SimpleHeuristicsPlayer}


def collect(args):
    teacher = PLAYERS[args.teacher](start_listening=False)
    pool = {n: c(start_listening=False) for n, c in PLAYERS.items()}
    weights = {"random": 0.2, "maxpower": 0.4, "heuristic": 0.4}
    names = list(weights)

    cur = str(np.random.choice(names, p=list(weights.values())))
    env = make_env(pool[cur], battle_format=args.format)
    base = env.env
    X, M, A, R = [], [], [], []
    record = {n: [] for n in names}
    t0 = time.time()

    for b in range(args.battles):
        cur = str(np.random.choice(names, p=list(weights.values())))
        env.opponent = pool[cur]
        obs, _ = env.reset()
        teacher.reset_battles()
        ep_obs, ep_mask, ep_act, ep_rew, ep_dec = [], [], [], [], []
        done = False
        while not done:
            battle = base.battle1
            mask = obs["action_mask"]
            legal = np.flatnonzero(mask)
            decision = len(legal) > 1
            if decision:
                teacher._battles[battle.battle_tag] = battle
                try:
                    order = teacher.choose_move(battle)
                    action = base.order_to_action(order, battle, fake=base._fake, strict=True)
                except Exception:
                    action = np.int64(np.random.choice(legal))
                if mask[int(action)] == 0:
                    action = np.int64(np.random.choice(legal))
            else:
                action = np.int64(legal[0]) if len(legal) else np.int64(0)
            ep_obs.append(obs["observation"])
            ep_mask.append(mask.astype(np.float32))
            ep_act.append(int(action))
            ep_dec.append(decision)
            obs, r, term, trunc, _ = env.step(action)
            ep_rew.append(r)
            done = term or trunc

        ret, g = np.zeros(len(ep_rew), dtype=np.float32), 0.0
        for t in reversed(range(len(ep_rew))):
            g = ep_rew[t] + args.gamma * g
            ret[t] = g
        for t, d in enumerate(ep_dec):
            if d:
                X.append(ep_obs[t]); M.append(ep_mask[t]); A.append(ep_act[t]); R.append(ret[t])
        record[cur].append(float(bool(base.battle1.won)))
        if (b + 1) % 100 == 0:
            per = " ".join(f"{n}={np.mean(v):.2f}({len(v)})" for n, v in record.items() if v)
            print(f"collected {b+1}/{args.battles} battles, {len(X)} decisions | teacher win rate: {per} | {time.time()-t0:.0f}s", flush=True)
    env.close()
    return (np.asarray(X, np.float32), np.asarray(M, np.float32),
            np.asarray(A, np.int64), np.asarray(R, np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", choices=PLAYERS, default="heuristic")
    ap.add_argument("--battles", type=int, default=3000)
    ap.add_argument("--format", default="gen9randombattle")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--data", default=None, help="reuse/save collected data (.npz)")
    args = ap.parse_args()

    if args.data and Path(args.data).exists():
        d = np.load(args.data)
        X, M, A, R = d["X"], d["M"], d["A"], d["R"]
        print(f"loaded {len(X)} decisions from {args.data}")
    else:
        X, M, A, R = collect(args)
        if args.data:
            np.savez_compressed(args.data, X=X, M=M, A=A, R=R)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(X)
    perm = np.random.permutation(n)
    n_val = max(n // 20, 1)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
    X_t, M_t, A_t, R_t = T(X), T(M), T(A, torch.int64), T(R)

    net = ActorCritic(OBS_DIM, M.shape[1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    Path(args.out).mkdir(exist_ok=True)

    def evaluate(idx):
        with torch.no_grad():
            dist, v = net(X_t[idx], M_t[idx])
            acc = (dist.probs.argmax(-1) == A_t[idx]).float().mean().item()
            vl = F.mse_loss(v, R_t[idx]).item()
        return acc, vl

    for ep in range(args.epochs):
        np.random.shuffle(tr_idx)
        tot = 0.0
        for s in range(0, len(tr_idx), args.batch):
            mb = torch.as_tensor(tr_idx[s : s + args.batch], device=device)
            dist, v = net(X_t[mb], M_t[mb])
            loss = F.nll_loss(dist.logits.log_softmax(-1), A_t[mb]) + 0.5 * F.mse_loss(v, R_t[mb])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(mb)
        acc, vl = evaluate(torch.as_tensor(val_idx, device=device))
        print(f"epoch {ep+1}/{args.epochs} train_loss={tot/len(tr_idx):.3f} val_action_acc={acc:.3f} val_value_mse={vl:.3f}", flush=True)
        torch.save(net.state_dict(), f"{args.out}/bc.pt")
    print(f"saved {args.out}/bc.pt")


if __name__ == "__main__":
    main()
