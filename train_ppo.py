"""Masked PPO against a curriculum of baseline opponents.

    python train_ppo.py --steps 200000 --opponent maxpower

Needs a local Showdown server on :8000 (see README).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from pokebot.env import make_env
from pokebot.encoder import OBS_DIM
from pokebot.model import ActorCritic

OPPONENTS = {
    "random": RandomPlayer,
    "maxpower": MaxBasePowerPlayer,
    "heuristic": SimpleHeuristicsPlayer,
}


def to_t(obs, device):
    o = torch.as_tensor(obs["observation"], dtype=torch.float32, device=device).unsqueeze(0)
    m = torch.as_tensor(obs["action_mask"], dtype=torch.float32, device=device).unsqueeze(0)
    return o, m


def gae(rew, val, done, last_val, gamma, lam):
    adv = np.zeros_like(rew)
    last = 0.0
    for t in reversed(range(len(rew))):
        nv = last_val if t == len(rew) - 1 else val[t + 1]
        nonterm = 1.0 - done[t]
        delta = rew[t] + gamma * nv * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--opponent", choices=[*OPPONENTS, "mixed"], default="maxpower",
                    help="'mixed' samples random/maxpower/heuristic (20/40/40) every battle")
    ap.add_argument("--format", default="gen9randombattle")
    ap.add_argument("--rollout", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--kl-coef", type=float, default=0.0,
                    help="penalty on KL(policy || anchor); keeps PPO close to the BC policy (try 0.2)")
    ap.add_argument("--anchor", default=None, help="frozen reference checkpoint (default: --resume)")
    ap.add_argument("--save-every", type=int, default=100_000,
                    help="also keep a snapshot step_<N>.pt this often, to pick the best one")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = {name: cls(start_listening=False) for name, cls in OPPONENTS.items()}
    weights = {"random": 0.2, "maxpower": 0.4, "heuristic": 0.4}

    def pick_opponent() -> str:
        if args.opponent != "mixed":
            return args.opponent
        return str(np.random.choice(list(weights), p=list(weights.values())))

    cur = pick_opponent()
    env = make_env(pool[cur], battle_format=args.format)
    by_opp = {name: [] for name in OPPONENTS}
    n_actions = env.action_space.n

    net = ActorCritic(OBS_DIM, n_actions).to(device)
    if args.resume:
        net.load_state_dict(torch.load(args.resume, map_location=device))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    Path(args.out).mkdir(exist_ok=True)

    ref = None
    if args.kl_coef > 0:
        anchor = args.anchor or args.resume
        assert anchor, "--kl-coef needs --anchor or --resume"
        ref = ActorCritic(OBS_DIM, n_actions).to(device)
        ref.load_state_dict(torch.load(anchor, map_location=device))
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    next_save = args.save_every

    obs, _ = env.reset()
    steps, battles, wins, ep_ret = 0, 0, 0, 0.0
    recent = []
    t0 = time.time()

    while steps < args.steps:
        buf = {k: [] for k in ("obs", "mask", "act", "logp", "rew", "val", "done")}
        for _ in range(args.rollout):
            o, m = to_t(obs, device)
            with torch.no_grad():
                dist, v = net(o, m)
                a = dist.sample()
                logp = dist.log_prob(a)
            nobs, r, term, trunc, info = env.step(np.int64(a.item()))
            done = term or trunc
            for k, x in zip(buf, (obs["observation"], obs["action_mask"], a.item(), logp.item(), r, v.item(), float(done))):
                buf[k].append(x)
            obs = nobs
            ep_ret += r
            steps += 1
            if done:
                battles += 1
                won = env.env.battle1.won if hasattr(env.env, "battle1") else ep_ret > 0
                recent.append(float(bool(won)))
                recent = recent[-100:]
                by_opp[cur] = (by_opp[cur] + [float(bool(won))])[-100:]
                cur = pick_opponent()
                env.opponent = pool[cur]
                obs, _ = env.reset()
                ep_ret = 0.0

        o, m = to_t(obs, device)
        with torch.no_grad():
            _, last_v = net(o, m)
        rew = np.array(buf["rew"], dtype=np.float32)
        val = np.array(buf["val"], dtype=np.float32)
        done = np.array(buf["done"], dtype=np.float32)
        adv, ret = gae(rew, val, done, last_v.item(), args.gamma, args.lam)

        T = lambda x, dt=torch.float32: torch.as_tensor(np.array(x), dtype=dt, device=device)
        b_obs, b_mask = T(buf["obs"]), T(buf["mask"])
        b_act, b_logp = T(buf["act"], torch.int64), T(buf["logp"])
        b_adv, b_ret = T(adv), T(ret)

        idx = np.arange(len(rew))
        kls = []
        for _ in range(args.epochs):
            np.random.shuffle(idx)
            for s in range(0, len(idx), args.minibatch):
                mb = torch.as_tensor(idx[s : s + args.minibatch], device=device)
                dist, v = net(b_obs[mb], b_mask[mb])
                ratio = (dist.log_prob(b_act[mb]) - b_logp[mb]).exp()
                a_ = b_adv[mb]
                a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                pg = -torch.min(ratio * a_, ratio.clamp(1 - args.clip, 1 + args.clip) * a_).mean()
                vl = 0.5 * (v - b_ret[mb]).pow(2).mean()
                loss = pg + 0.5 * vl - args.ent * dist.entropy().mean()
                if ref is not None:
                    with torch.no_grad():
                        ref_dist, _ = ref(b_obs[mb], b_mask[mb])
                    kl = (dist.probs * (dist.logits - ref_dist.logits)).masked_fill(b_mask[mb] < 0.5, 0.0).sum(-1).mean()
                    loss = loss + args.kl_coef * kl
                    kls.append(kl.item())
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        wr = np.mean(recent) if recent else float("nan")
        per = " ".join(f"{n}={np.mean(v):.2f}" for n, v in by_opp.items() if v)
        kl_txt = f" kl={np.mean(kls):.3f}" if kls else ""
        print(f"steps={steps} battles={battles} winrate(last100)={wr:.2f} [{per}]{kl_txt} sps={steps/(time.time()-t0):.0f}", flush=True)
        torch.save(net.state_dict(), f"{args.out}/latest.pt")
        if args.save_every and steps >= next_save:
            torch.save(net.state_dict(), f"{args.out}/step_{steps}.pt")
            next_save += args.save_every


if __name__ == "__main__":
    main()
