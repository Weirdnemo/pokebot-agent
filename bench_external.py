"""Play one of our bots against an external bot (e.g. Foul Play) running on the same
local Showdown server, waiting for challenges.

    # terminal 1 (our bot waits for the external bot to challenge it):
    python bench_external.py --opponent FoulPlayBot --bot heuristic --n 50
    python bench_external.py --opponent FoulPlayBot --bot ckpt --ckpt checkpoints/bc.pt --n 50
    # terminal 2: start the external bot AFTER the line above is running, e.g. Foul Play with
    #   --bot-mode challenge_user --user-to-challenge OurBot --run-count 50

Reports wins/n and a 95% Wilson interval, because with n in the tens the noise is large.
"""
from __future__ import annotations

import argparse
import asyncio
import math

import numpy as np
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer

BOTS = {"random": RandomPlayer, "maxpower": MaxBasePowerPlayer, "heuristic": SimpleHeuristicsPlayer}


def wilson(w: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def make_ckpt_player(path: str, fmt: str, acct=None):
    import torch

    from pokebot.encoder import encode_battle, OBS_DIM
    from pokebot.env import PokemonEnv
    from pokebot.model import ActorCritic

    n_actions = 26 if "gen9" in fmt else 10
    net = ActorCritic(OBS_DIM, n_actions)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()

    class CkptPlayer(Player):
        def choose_move(self, battle):
            o = torch.as_tensor(encode_battle(battle), dtype=torch.float32).unsqueeze(0)
            m = torch.as_tensor(PokemonEnv.get_action_mask(battle), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                dist, _ = net(o, m)
            a = np.int64(dist.probs.argmax(-1).item())
            return PokemonEnv.action_to_order(a, battle, strict=False)

    return CkptPlayer(battle_format=fmt, account_configuration=acct)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", required=True, help="username of the external bot")
    ap.add_argument("--bot", choices=[*BOTS, "ckpt"], default="heuristic")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--format", default="gen9randombattle")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--mode", choices=["challenge", "accept"], default="accept",
                    help="accept: wait for the external bot to challenge us (start us first, "
                         "then start it with --bot-mode challenge_user --user-to-challenge OurBot). "
                         "challenge: we challenge it (start it first, in accept_challenge mode)")
    ap.add_argument("--name", default="OurBot", help="our account name (needed in accept mode)")
    args = ap.parse_args()

    from poke_env import AccountConfiguration

    acct = AccountConfiguration(args.name, None)
    if args.bot == "ckpt":
        assert args.ckpt, "--ckpt required"
        player = make_ckpt_player(args.ckpt, args.format, acct)
    else:
        player = BOTS[args.bot](battle_format=args.format, account_configuration=acct)

    async def announce_ready():
        # Challenges sent before we are logged in are silently lost, and the challenger then
        # waits forever. So tell the user when it is safe to start the external bot.
        while not player.ps_client.logged_in.is_set():
            await asyncio.sleep(0.25)
        if args.mode == "accept":
            print(f"[ready] {args.name} is logged in and waiting for {args.n} challenge(s) from "
                  f"{args.opponent}. Start the external bot NOW.", flush=True)
        else:
            print(f"[ready] {args.name} is logged in; challenging {args.opponent} (it must already "
                  f"be running and waiting).", flush=True)

        # Progress: announce each battle when it starts (with a URL you can open to watch it)
        # and each time one finishes.
        seen, last_done = set(), -1
        while True:
            await asyncio.sleep(2)
            for tag, b in list(player.battles.items()):
                if tag not in seen:
                    seen.add(tag)
                    print(f"[battle] started {tag} -> watch at http://localhost:8000/{tag}", flush=True)
            done = player.n_finished_battles
            if done != last_done:
                last_done = done
                if done:
                    print(f"[progress] {done}/{args.n} finished, {player.n_won_battles} won", flush=True)

    ready_task = asyncio.create_task(announce_ready())

    if args.mode == "challenge":
        await player.send_challenges(args.opponent, n_challenges=args.n)
    else:
        await player.accept_challenges(args.opponent, args.n)
    ready_task.cancel()
    w, n = player.n_won_battles, player.n_finished_battles
    lo, hi = wilson(w, n)
    print(f"{args.bot} vs {args.opponent}: {w}/{n} = {w / max(n, 1):.2f}  (95% CI {lo:.2f}-{hi:.2f})")


if __name__ == "__main__":
    asyncio.run(main())
