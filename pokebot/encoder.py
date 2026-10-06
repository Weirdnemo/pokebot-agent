"""Flat numeric encoding of a poke-env singles Battle.

Deliberately simple v0: everything is a fixed-size float vector so a plain MLP
works. Known upgrade path: replace the one-hot / stat features with learned
embeddings for species, moves, items and abilities, and feed the 12 mons as a
set to a transformer.
"""
from __future__ import annotations

import numpy as np
from poke_env.battle import Battle, Move, MoveCategory, Pokemon, PokemonType, SideCondition, Status, Weather

from .damage import est_speed, move_damage_frac, stab_threat

TYPES = list(PokemonType)  # includes THREE_QUESTION_MARKS; fine as padding
STATUSES = list(Status)
WEATHERS = list(Weather)
BOOST_KEYS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
HAZARDS = [
    SideCondition.STEALTH_ROCK,
    SideCondition.SPIKES,
    SideCondition.TOXIC_SPIKES,
    SideCondition.STICKY_WEB,
]
SCREENS = [SideCondition.REFLECT, SideCondition.LIGHT_SCREEN, SideCondition.AURORA_VEIL, SideCondition.TAILWIND]

N_TEAM = 6
N_MOVES = 4

MOVE_DIM = 3 + len(TYPES) + 3 + 2  # bp, acc, pp, type, category, priority, effectiveness
MON_DIM = (
    1  # hp fraction
    + 1  # fainted
    + 1  # is active
    + len(STATUSES)
    + len(TYPES)  # type multi-hot
    + 6  # base stats / 255
    + 1  # terastallized
    + 1  # known (slot filled)
)
BOOST_DIM = len(BOOST_KEYS)
FIELD_DIM = len(WEATHERS) + 2 * (len(HAZARDS) + len(SCREENS))


def _onehot(idx: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


def encode_move(move: Move | None, defender: Pokemon | None) -> np.ndarray:
    if move is None:
        return np.zeros(MOVE_DIM, dtype=np.float32)
    bp = (move.base_power or 0) / 150.0
    acc = 1.0 if move.accuracy is True else float(move.accuracy or 0.0)
    pp = move.current_pp / max(move.max_pp, 1)
    t = _onehot(TYPES.index(move.type), len(TYPES)) if move.type in TYPES else np.zeros(len(TYPES), np.float32)
    cat = _onehot(list(MoveCategory).index(move.category), 3)
    prio = move.priority / 5.0
    eff = 0.0
    if defender is not None and move.type is not None:
        try:
            eff = float(defender.damage_multiplier(move)) / 4.0
        except Exception:
            eff = 0.0
    return np.concatenate([[bp, acc, pp], t, cat, [prio, eff]]).astype(np.float32)


def encode_mon(mon: Pokemon | None, is_active: bool) -> np.ndarray:
    if mon is None:
        return np.zeros(MON_DIM, dtype=np.float32)
    status = _onehot(STATUSES.index(mon.status), len(STATUSES)) if mon.status is not None else np.zeros(len(STATUSES), np.float32)
    types = np.zeros(len(TYPES), dtype=np.float32)
    for t in mon.types:
        if t is not None and t in TYPES:
            types[TYPES.index(t)] = 1.0
    stats = np.array([(mon.base_stats or {}).get(k, 0) for k in ("hp", "atk", "def", "spa", "spd", "spe")], dtype=np.float32) / 255.0
    return np.concatenate(
        [
            [mon.current_hp_fraction, float(mon.fainted), float(is_active)],
            status,
            types,
            stats,
            [float(getattr(mon, "is_terastallized", False)), 1.0],
        ]
    ).astype(np.float32)


def encode_boosts(mon: Pokemon | None) -> np.ndarray:
    if mon is None:
        return np.zeros(BOOST_DIM, dtype=np.float32)
    return np.array([mon.boosts.get(k, 0) / 6.0 for k in BOOST_KEYS], dtype=np.float32)


def encode_side(conds: dict, keys: list) -> np.ndarray:
    out = np.zeros(len(keys), dtype=np.float32)
    for i, k in enumerate(keys):
        if k in conds:
            # spikes / toxic spikes stack up to 3 layers; others are 0/1
            out[i] = min(conds[k], 3) / 3.0 if k in (SideCondition.SPIKES, SideCondition.TOXIC_SPIKES) else 1.0
    return out


def encode_battle(battle: Battle) -> np.ndarray:
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon

    own = [encode_mon(m, m is me) for m in list(battle.team.values())[:N_TEAM]]
    own += [encode_mon(None, False)] * (N_TEAM - len(own))

    # opponent: only the mons we have seen; unseen slots stay zero (known=0)
    foe = [encode_mon(m, m is opp) for m in list(battle.opponent_team.values())[:N_TEAM]]
    foe += [encode_mon(None, False)] * (N_TEAM - len(foe))

    moves = list(me.moves.values())[:N_MOVES] if me is not None else []
    moves += [None] * (N_MOVES - len(moves))
    move_feats = [encode_move(m, opp) for m in moves]

    weather = np.zeros(len(WEATHERS), dtype=np.float32)
    for w in battle.weather:
        weather[WEATHERS.index(w)] = 1.0

    field = np.concatenate(
        [
            weather,
            encode_side(battle.side_conditions, HAZARDS + SCREENS),
            encode_side(battle.opponent_side_conditions, HAZARDS + SCREENS),
        ]
    )

    extra = np.array(
        [
            float(battle.can_tera),
            battle.turn / 50.0,
            len([m for m in battle.team.values() if not m.fainted]) / 6.0,
            (6 - len([m for m in battle.opponent_team.values() if m.fainted])) / 6.0,
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        own
        + foe
        + move_feats
        + [encode_boosts(me), encode_boosts(opp), field, extra, encode_matchup(battle)]
    ).astype(np.float32)


def _clip(x: float, hi: float = 2.0) -> float:
    return float(min(max(x, 0.0), hi)) / hi


def encode_matchup(battle: Battle) -> np.ndarray:
    """v2 features: what the moves actually do, what the opponent has shown, bench options.

    All damage numbers are expected damage as a fraction of the defender's max HP,
    clipped to [0, 2] and rescaled to [0, 1] (so 0.5 means 'about a full KO').
    """
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    out: list[float] = []

    # 1) our active moves: estimated damage, and damage relative to the opp's CURRENT hp
    my_moves = list(me.moves.values())[:N_MOVES] if me is not None else []
    my_moves += [None] * (N_MOVES - len(my_moves))
    for mv in my_moves:
        d = move_damage_frac(mv, me, opp, True) if (mv is not None and me and opp) else 0.0
        cur = max(opp.current_hp_fraction, 1e-3) if opp is not None else 1.0
        out += [_clip(d), _clip(d / cur), float(mv is not None and mv.type in (me.types if me else []))]

    # 2) opponent's revealed moves: shape + estimated damage to our active
    opp_moves = list(opp.moves.values())[:N_MOVES] if opp is not None else []
    opp_moves += [None] * (N_MOVES - len(opp_moves))
    for mv in opp_moves:
        if mv is None or me is None or opp is None:
            out += [0.0] * (MOVE_DIM + 2)
            continue
        d = move_damage_frac(mv, opp, me, False)
        cur = max(me.current_hp_fraction, 1e-3)
        out += list(encode_move(mv, me)) + [_clip(d), _clip(d / cur)]

    # 3) opponent STAB threat against each of our six mons (physical / special guess)
    team = list(battle.team.values())[:N_TEAM]
    for i in range(N_TEAM):
        if i < len(team) and opp is not None:
            p, s = stab_threat(opp, team[i], False)
            out += [_clip(p), _clip(s)]
        else:
            out += [0.0, 0.0]

    # 4) bench options: each own mon's moves vs the opp active (damage, effectiveness, stab)
    for i in range(N_TEAM):
        m = team[i] if i < len(team) else None
        mvs = list(m.moves.values())[:N_MOVES] if m is not None else []
        mvs += [None] * (N_MOVES - len(mvs))
        best = 0.0
        for mv in mvs:
            if mv is None or m is None or opp is None:
                out += [0.0, 0.0]
                continue
            d = move_damage_frac(mv, m, opp, True)
            best = max(best, d)
            try:
                eff = float(opp.damage_multiplier(mv)) / 4.0 if mv.base_power else 0.0
            except Exception:
                eff = 0.0
            out += [_clip(d), eff]
        out.append(_clip(best))

    # 5) speed: who moves first (log2 ratio in [-2, 2] -> [0, 1]), plus raw faster flag
    if me is not None and opp is not None:
        ratio = est_speed(me, True) / max(est_speed(opp, False), 1.0)
        out += [(float(np.clip(np.log2(max(ratio, 1e-3)), -2.0, 2.0)) + 2.0) / 4.0, float(ratio > 1.0)]
    else:
        out += [0.5, 0.0]

    return np.asarray(out, dtype=np.float32)


MATCHUP_DIM = (
    N_MOVES * 3  # own active moves
    + N_MOVES * (MOVE_DIM + 2)  # opponent revealed moves
    + N_TEAM * 2  # opp STAB threat per own mon
    + N_TEAM * (N_MOVES * 2 + 1)  # bench moves
    + 2  # speed
)

OBS_DIM = (
    2 * N_TEAM * MON_DIM + N_MOVES * MOVE_DIM + 2 * BOOST_DIM + FIELD_DIM + 4 + MATCHUP_DIM
)
