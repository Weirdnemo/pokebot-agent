"""Set-pool beliefs about the opponent's hidden moves and Tera type (random battles).

Why this exists: the E13 error analysis showed our network usually knows WHEN Foul Play switches
but picks the wrong mon (about 35% right). Choosing the switch target means asking, for each of
our six mons, "how badly would the opponent's active hurt it?". The v2 encoder only had a generic
STAB guess for that. In random battles every species has a short list of roles (Showdown's
`data/random-battles/gen9/sets.json`), each with a move pool, so after a few turns the opponent's
possible sets are quite constrained. This module turns that into features, aligned to OUR six team
slots (same order as the switch actions 0-5).

Model (an approximation, written down so it can be criticised):
  * Roles of the opponent's active species that contain every revealed move (and its revealed
    ability, if known) are equally likely. If none does, fall back to the roles that contain the
    most revealed moves, then to all roles.
  * Within a role, a set is 4 moves; revealed moves are in it, and each unrevealed pool move is in it
    independently with probability min(1, free_slots / unrevealed_pool_size). (The real generator is
    not independent; this is a cheap stand-in.)
  * For each of our mons: worst-case hit = max over consistent roles of the strongest move in
    (revealed + pool); expected hit = mean over roles of E[strongest move present].

Needs the sets file. Copy it once:
    cp <pokemon-showdown>/data/random-battles/gen9/sets.json  <pokebot>/data/gen9_random_sets.json
or set POKEBOT_SETS=/path/to/sets.json.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
from poke_env.battle import Battle, Move, PokemonType

from .damage import est_speed, move_damage_frac

N_TEAM = 6
TYPES = list(PokemonType)
N_T = len(TYPES)
PER_SLOT = 6
BELIEF_DIM = N_TEAM * PER_SLOT + 2 + N_T

_SETS: dict | None = None
_MOVES: dict = {}


def norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _candidate_paths():
    env = os.environ.get("POKEBOT_SETS")
    if env:
        yield Path(env)
    here = Path(__file__).resolve().parent.parent
    yield here / "data" / "gen9_random_sets.json"
    yield Path.cwd() / "data" / "gen9_random_sets.json"
    for base in (here.parent, here, Path.home(), Path.home() / "Reinforcement_Learning"):
        yield base / "pokemon-showdown" / "data" / "random-battles" / "gen9" / "sets.json"


def load_sets() -> dict:
    global _SETS
    if _SETS is None:
        for p in _candidate_paths():
            if p.is_file():
                _SETS = json.loads(p.read_text())
                break
        else:
            raise FileNotFoundError(
                "Random-battle set data not found (needed by the v4 belief features). Copy it with:\n"
                "  cp <pokemon-showdown>/data/random-battles/gen9/sets.json data/gen9_random_sets.json\n"
                "or set POKEBOT_SETS=/path/to/sets.json"
            )
    return _SETS


def _move(move_id: str) -> Move | None:
    key = norm(move_id)
    if key not in _MOVES:
        try:
            _MOVES[key] = Move(key, gen=9)
        except Exception:
            _MOVES[key] = None
    return _MOVES[key]


def consistent_roles(mon, sets: dict | None = None) -> list[dict]:
    """Roles of `mon`'s species compatible with what has been revealed (see module docstring)."""
    sets = sets if sets is not None else load_sets()
    entry = sets.get(norm(mon.species)) or sets.get(norm(getattr(mon, "base_species", ""))) or None
    if not entry:
        return []
    roles = entry["sets"]
    revealed = [norm(k) for k in mon.moves.keys()]
    pools = [{norm(m) for m in r["movepool"]} for r in roles]
    counts = [sum(m in p for m in revealed) for p in pools]
    full = [i for i, c in enumerate(counts) if c == len(revealed)]
    keep = full or [i for i, c in enumerate(counts) if c == max(counts)] if counts else []
    ability = norm(mon.ability) if getattr(mon, "ability", None) else ""
    if ability:
        by_ab = [i for i in keep if ability in {norm(a) for a in roles[i].get("abilities", [])}]
        keep = by_ab or keep
    return [roles[i] for i in keep]


def _expected_max(values: list[float], rev_max: float, p: float) -> float:
    """E[max damage] when each value in `values` is independently present with probability p
    and `rev_max` is always present."""
    acc, remaining = 0.0, 1.0
    for d in sorted(values, reverse=True):
        if d <= rev_max:
            break
        acc += remaining * p * d
        remaining *= 1.0 - p
    return acc + remaining * rev_max


def _clip(x: float, hi: float = 2.0) -> float:
    return float(min(max(x, 0.0), hi)) / hi


def encode_belief(battle: Battle) -> np.ndarray:
    out = np.zeros(BELIEF_DIM, dtype=np.float32)
    opp = battle.opponent_active_pokemon
    if opp is None:
        return out
    roles = consistent_roles(opp)
    revealed_moves = [m for m in opp.moves.values()]
    n_rev = len(revealed_moves)
    team = list(battle.team.values())[:N_TEAM]

    # pool of candidate unrevealed moves per role, with the chance each one is in the set
    rev_ids = {norm(k) for k in opp.moves.keys()}
    role_pools = []
    for r in roles:
        pool = [m for m in (_move(x) for x in r["movepool"]) if m is not None and norm(m.id) not in rev_ids]
        free = max(0, 4 - n_rev)
        p = min(1.0, free / len(pool)) if pool else 0.0
        role_pools.append((pool, p))

    for i in range(N_TEAM):
        if i >= len(team):
            continue
        mon = team[i]
        if mon is None or mon.fainted:
            continue
        cur = max(mon.current_hp_fraction, 0.05)
        rev_max = max([move_damage_frac(mv, opp, mon, False) for mv in revealed_moves] or [0.0])
        worst, exp = rev_max, rev_max
        if role_pools:
            ws, es = [], []
            for pool, p in role_pools:
                vals = [move_damage_frac(m, opp, mon, False) for m in pool]
                ws.append(max([rev_max] + vals))
                es.append(_expected_max(vals, rev_max, p))
            worst, exp = max(ws), float(np.mean(es))
        faster = float(est_speed(mon, True) > est_speed(opp, False))
        j = i * PER_SLOT
        out[j:j + PER_SLOT] = [
            _clip(worst), _clip(exp), _clip(exp / cur), _clip(rev_max), faster, float(worst >= cur),
        ]

    j = N_TEAM * PER_SLOT
    out[j] = len(roles) / 3.0
    out[j + 1] = n_rev / 4.0
    # Tera-type belief: uniform over consistent roles, then uniform over each role's Tera types
    tt = np.zeros(N_T, dtype=np.float32)
    for r in roles:
        types = r.get("teraTypes") or []
        for t in types:
            try:
                tt[TYPES.index(PokemonType[norm(t).upper()])] += 1.0 / (len(types) * len(roles))
            except (KeyError, ValueError):
                pass
    out[j + 2:j + 2 + N_T] = tt
    return out
