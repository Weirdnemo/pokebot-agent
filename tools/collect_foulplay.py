"""Record Foul Play's decisions together with OUR observation of the same game state.

One process, one Showdown account (default "FoulPlayBot"). A poke-env Player owns the
websocket connection and keeps its own Battle object up to date. Every raw room message
it receives is also forwarded, unchanged and AFTER poke-env has processed it, to Foul
Play's own battle logic (fp.run_battle.pokemon_battle), which runs its normal MCTS
search. Foul Play never talks to the server directly: its "send" calls are intercepted.
When Foul Play picks a move we log

    obs      our encoder's vector for the state (pokebot.encoder.encode_battle)
    mask     legal-action mask in our 26-action space
    act      Foul Play's chosen action, mapped into our action space
    policy   Foul Play's whole search policy (visit shares over its options, summed over
             the sampled opponent worlds), mapped into our action space and normalised
    + battle id, turn, number of legal actions, and the final result of the battle

and then send Foul Play's choice to the server. The opponent is a built-in poke-env
bot in the same process, so there is no second terminal and no login patch needed.

Run in the Foul Play env (poke2), which also needs poke-env 0.16.1 and numpy:

    pip install "poke-env==0.16.1" numpy        # once, in poke2
    node pokemon-showdown start --no-security   # in the Showdown folder
    cd ~/Reinforcement_Learning/pokebot
    python tools/collect_foulplay.py --foul-play-dir ./foul-play --battles 20 --out data/fp_test.npz

Foul Play's search settings are passed through (--search-time-ms, --search-parallelism)
and are stored in the output file; keep them the same as in your benchmark runs unless
you are deliberately changing the teacher's strength.

Licence note: this imports Foul Play (GPL-3.0) as a library and monkey-patches two of
its functions at run time. Its source is not copied into this repository.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

N_ACTIONS = 26  # poke-env SinglesEnv action space for gen9 (6 switches + 4 moves x 5 modes)


def norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# ---------------------------------------------------------------------------------------
# Foul Play side: configuration, and two thin wrappers that expose what the search decided
# ---------------------------------------------------------------------------------------
_tls = threading.local()      # which battle the search thread is working on
LAST_CHOICE: dict = {}        # battle_tag -> Foul Play's chosen option, e.g. "earthquake-tera"
LAST_POLICY: dict = {}        # battle_tag -> {option: visit share}, over ALL options


def setup_foul_play(args):
    sys.path.insert(0, str(Path(args.foul_play_dir).resolve()))
    os.chdir(Path(args.foul_play_dir).resolve())  # Foul Play uses a few relative paths
    sys.argv = [
        "foul-play",
        "--websocket-uri", args.websocket_uri,
        "--ps-username", args.fp_name,
        "--bot-mode", "accept_challenge",  # unused here, but required by its argument parser
        "--pokemon-format", args.format,
        "--search-time-ms", str(args.search_time_ms),
        "--search-parallelism", str(args.search_parallelism),
        "--log-level", "WARNING",
    ]
    from fp.config import FoulPlayConfig, init_logging
    from fp.data.mods.apply_mods import apply_mods

    FoulPlayConfig.configure()
    init_logging("WARNING", False)
    apply_mods(FoulPlayConfig.format_spec)

    import fp.modes.base as fp_base
    import fp.search.main as fp_main

    orig_find = fp_base.find_best_move          # name used by async_pick_move
    orig_select = fp_main.select_move_from_mcts_results  # name used by find_best_move

    def find_best_move_logged(battle):
        _tls.tag = battle.battle_tag
        choice = orig_find(battle)
        LAST_CHOICE[battle.battle_tag] = choice
        return choice

    def select_logged(mcts_results):
        policy = {}
        for res, chance, _idx in mcts_results:
            for opt in res.side_one:
                policy[opt.move_choice] = policy.get(opt.move_choice, 0.0) + chance * (
                    opt.visits / res.total_visits
                )
        LAST_POLICY[getattr(_tls, "tag", None)] = policy
        return orig_select(mcts_results)

    fp_base.find_best_move = find_best_move_logged
    fp_main.select_move_from_mcts_results = select_logged

    from fp.run_battle import pokemon_battle
    return pokemon_battle, FoulPlayConfig


class QueueWS:
    """Stands in for Foul Play's PSWebsocketClient for ONE battle room."""

    def __init__(self, recorder: "FPRecorder", tag: str):
        self.q: asyncio.Queue = asyncio.Queue()
        self.rec = recorder
        self.tag = tag

    async def receive_message(self) -> str:
        return await self.q.get()

    async def send_message(self, room, message_list):
        await self.rec.on_fp_send(room or self.tag, message_list)

    async def join_room(self, room_name):
        pass

    async def save_replay(self, battle_tag):
        pass

    async def leave_battle(self, battle_tag):
        pass  # poke-env already leaves the room when the battle ends

    async def close(self):
        pass


# ---------------------------------------------------------------------------------------
# poke-env side
# ---------------------------------------------------------------------------------------
def make_recorder_class():
    from poke_env.player import Player

    from pokebot.encoder import encode_battle
    from pokebot.env import PokemonEnv

    class FPRecorder(Player):
        def __init__(self, *a, pokemon_battle_fn=None, fp_format="gen9randombattle", **kw):
            super().__init__(*a, **kw)
            self._fp_battle_fn = pokemon_battle_fn
            self._fp_format = fp_format
            self._fp_ws: dict[str, QueueWS] = {}
            self._fp_tasks: dict[str, asyncio.Task] = {}
            self.rows: list[dict] = []
            self.unmapped = 0
            self.fp_errors = 0

        # poke-env must NOT choose moves: Foul Play does. Keep the method abstract-safe.
        def choose_move(self, battle):
            return self.choose_default_move()

        async def _handle_battle_request(self, battle, maybe_default_order=False):
            return  # state tracking only

        async def _handle_battle_message(self, split_messages):
            await super()._handle_battle_message(split_messages)  # poke-env state first
            tag = split_messages[0][0][1:]
            if not tag.startswith("battle"):
                return
            raw = "\n".join("|".join(m) for m in split_messages)
            ws = self._fp_ws.get(tag)
            if ws is None:
                if not (len(split_messages) > 1 and len(split_messages[1]) > 1
                        and split_messages[1][1] == "init"):
                    return  # joined mid-battle: cannot follow it
                ws = QueueWS(self, tag)
                self._fp_ws[tag] = ws
                self._fp_tasks[tag] = asyncio.create_task(self._run_fp(ws, tag))
            ws.q.put_nowait(raw)

        async def _run_fp(self, ws, tag):
            try:
                await self._fp_battle_fn(ws, self._fp_format, None)
            except Exception as e:  # keep collecting even if one battle breaks
                self.fp_errors += 1
                print(f"[warn] Foul Play failed in {tag}: {type(e).__name__}: {e}", flush=True)

        # --- everything Foul Play wants to send passes through here ---
        async def on_fp_send(self, room: str, message_list: list):
            first = message_list[0] if message_list else ""
            if first in ("gg", "hf") or first.startswith("/timer"):
                return  # chat / timer: not part of the game
            if first.startswith("/choose") or first.startswith("/switch"):
                self._record(room)
            m2 = message_list[1] if len(message_list) > 1 else None
            await self.ps_client.send_message(first, room, m2)

        def _legal_actions(self, battle):
            mask = np.array(PokemonEnv.get_action_mask(battle), dtype=np.int8)
            return mask, [a for a in range(N_ACTIONS) if mask[a]]

        def _map_choice(self, battle, choice: str, legal) -> int | None:
            """Foul Play option string -> our action index (or None)."""
            tera = choice.endswith("-tera")
            base = choice.removesuffix("-tera").removesuffix("-mega")
            for a in legal:
                try:
                    order = PokemonEnv.action_to_order(np.int64(a), battle, strict=False)
                except Exception:
                    continue
                msg = order.message
                if choice.startswith("switch "):
                    if msg.startswith("/choose switch ") and a < 6:
                        mon = list(battle.team.values())[a]
                        want = norm(choice[len("switch "):])
                        if want in (norm(mon.species), norm(mon.base_species), norm(mon.name)):
                            return a
                else:
                    want = f"/choose move {base}" + (" terastallize" if tera else "")
                    if msg == want:
                        return a
            return None

        def _record(self, tag: str):
            battle = self.battles.get(tag)
            choice = LAST_CHOICE.pop(tag, None)
            policy = LAST_POLICY.pop(tag, None)
            if battle is None or choice is None or policy is None:
                return  # e.g. forced single option: nothing was searched
            mask, legal = self._legal_actions(battle)
            act = self._map_choice(battle, choice, legal)
            if act is None:
                self.unmapped += 1
                return
            pol = np.zeros(N_ACTIONS, dtype=np.float32)
            for k, v in policy.items():
                a = self._map_choice(battle, k, legal)
                if a is not None:
                    pol[a] += v
            if pol.sum() <= 0:
                self.unmapped += 1
                return
            pol /= pol.sum()
            self.rows.append(dict(
                obs=encode_battle(battle).astype(np.float32), mask=mask, act=act, policy=pol,
                n_legal=len(legal), turn=int(battle.turn), tag=tag,
            ))

    return FPRecorder


def save(path: Path, rows, results, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    tags = sorted({r["tag"] for r in rows})
    bid = {t: i for i, t in enumerate(tags)}
    np.savez_compressed(
        path,
        obs=np.stack([r["obs"] for r in rows]),
        mask=np.stack([r["mask"] for r in rows]),
        act=np.array([r["act"] for r in rows], dtype=np.int16),
        policy=np.stack([r["policy"] for r in rows]),
        n_legal=np.array([r["n_legal"] for r in rows], dtype=np.int16),
        turn=np.array([r["turn"] for r in rows], dtype=np.int16),
        battle=np.array([bid[r["tag"]] for r in rows], dtype=np.int32),
        # +1 if Foul Play won that battle, -1 if it lost, 0 if unknown/tie
        outcome=np.array([results.get(r["tag"], 0) for r in rows], dtype=np.int8),
        meta=np.array(str(meta)),
    )


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--foul-play-dir", required=True, help="path of your foul-play clone")
    ap.add_argument("--battles", type=int, default=20)
    ap.add_argument("--opponent", choices=["random", "maxpower", "heuristic"], default="heuristic")
    ap.add_argument("--format", default="gen9randombattle")
    ap.add_argument("--websocket-uri", default="ws://localhost:8000/showdown/websocket")
    ap.add_argument("--fp-name", default="FoulPlayBot")
    ap.add_argument("--opp-name", default="OurBot")
    ap.add_argument("--search-time-ms", type=int, default=100)
    ap.add_argument("--search-parallelism", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=10, help="save after this many battles")
    ap.add_argument("--out", default="data/fp_data.npz")
    args = ap.parse_args()

    # our package must be importable even though we chdir into the Foul Play folder
    here = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(here))
    out = Path(args.out)
    if not out.is_absolute():
        out = here / out

    pokemon_battle_fn, FoulPlayConfig = setup_foul_play(args)

    from poke_env import AccountConfiguration
    from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

    opp_cls = {"random": RandomPlayer, "maxpower": MaxBasePowerPlayer,
               "heuristic": SimpleHeuristicsPlayer}[args.opponent]
    Rec = make_recorder_class()
    rec = Rec(account_configuration=AccountConfiguration(args.fp_name, None),
              battle_format=args.format, pokemon_battle_fn=pokemon_battle_fn, log_level=40,
              fp_format=args.format)
    opp = opp_cls(account_configuration=AccountConfiguration(args.opp_name, None),
                  battle_format=args.format, log_level=40)

    meta = dict(search_time_ms=args.search_time_ms, search_parallelism=args.search_parallelism,
                opponent=args.opponent, format=args.format, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"[collect] {args.battles} battles, Foul Play {args.search_time_ms} ms x "
          f"{args.search_parallelism} vs {args.opponent}; saving to {out}", flush=True)

    done, t0 = 0, time.time()
    while done < args.battles:
        n = min(args.chunk, args.battles - done)
        await rec.battle_against(opp, n_battles=n)
        done += n
        results = {t: (1 if b.won else -1 if b.won is False else 0) for t, b in rec.battles.items()
                   if b.finished}
        if rec.rows:
            save(out, rec.rows, results, meta)
        w = sum(1 for v in results.values() if v == 1)
        print(f"[collect] {done}/{args.battles} battles, {len(rec.rows)} decisions, "
              f"Foul Play won {w}/{len(results)}, unmapped={rec.unmapped}, "
              f"fp_errors={rec.fp_errors}, {time.time() - t0:.0f}s", flush=True)
    print(f"[done] saved {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
