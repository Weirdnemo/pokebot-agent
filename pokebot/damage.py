"""Cheap damage / speed estimates used as observation features.

poke-env ships a full Gen 9 calculator, but it asserts that both sides' stats are
known, which is never true for the opponent mid-battle. This module uses the
standard damage formula with stats estimated from base stats (random-battle
spread: 31 IVs, 85 EVs, neutral nature), so it works for any Pokémon on the field.
"""
from __future__ import annotations

import math

from poke_env.battle import MoveCategory, Pokemon, PokemonType, Status


def boost_mult(stage: int) -> float:
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def est_stat(mon: Pokemon, key: str, own: bool) -> float:
    """Actual stat for our mons when known, otherwise an estimate from base stats."""
    if own:
        if key == "hp" and getattr(mon, "max_hp", 0) and mon.max_hp > 1:
            return float(mon.max_hp)
        v = (mon.stats or {}).get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    base = (mon.base_stats or {}).get(key, 80)
    level = mon.level or 100
    core = math.floor((2 * base + 31 + 21) * level / 100)
    return float(core + level + 10) if key == "hp" else float(core + 5)


def est_speed(mon: Pokemon, own: bool) -> float:
    s = est_stat(mon, "spe", own) * boost_mult(mon.boosts.get("spe", 0))
    if mon.status == Status.PAR:
        s *= 0.5
    return s


def raw_damage_frac(
    bp: float,
    mtype: PokemonType | None,
    category: MoveCategory,
    attacker: Pokemon,
    defender: Pokemon,
    attacker_own: bool,
    hits: float = 1.0,
    stab_mod: float | None = None,
    defender_types: list | None = None,
) -> float:
    """Expected damage (average roll, no crit) as a fraction of defender max HP.

    `stab_mod` overrides the STAB multiplier and `defender_types` the defender's types
    (both used to ask "what if it terastallized?")."""
    if bp <= 0 or category == MoveCategory.STATUS:
        return 0.0
    physical = category == MoveCategory.PHYSICAL
    a_key, d_key = ("atk", "def") if physical else ("spa", "spd")
    atk = est_stat(attacker, a_key, attacker_own) * boost_mult(attacker.boosts.get(a_key, 0))
    dfn = est_stat(defender, d_key, not attacker_own) * boost_mult(defender.boosts.get(d_key, 0))
    level = attacker.level or 100
    base = ((2 * level / 5 + 2) * bp * atk / max(dfn, 1.0)) / 50 + 2
    mod = 0.925  # average of the 0.85-1.0 random roll
    if stab_mod is not None:
        mod *= stab_mod
    elif mtype is not None and mtype in attacker.types:
        mod *= 1.5
    if mtype is not None:
        try:
            if defender_types is not None:
                mod *= type_mult(mtype, defender_types)
            else:
                mod *= float(defender.damage_multiplier(mtype))
        except Exception:
            pass
    if physical and attacker.status == Status.BRN:
        mod *= 0.5
    hp = est_stat(defender, "hp", not attacker_own)
    return float(base * mod * hits / max(hp, 1.0))


def move_damage_frac(
    move, attacker: Pokemon, defender: Pokemon, attacker_own: bool,
    stab_mod: float | None = None, defender_types: list | None = None,
) -> float:
    if move is None or defender is None or attacker is None:
        return 0.0
    hits = float(getattr(move, "expected_hits", 1) or 1)
    return raw_damage_frac(
        float(move.base_power or 0), move.type, move.category, attacker, defender, attacker_own, hits,
        stab_mod=stab_mod, defender_types=defender_types,
    )


def type_mult(mtype: PokemonType, types: list, gen: int = 9) -> float:
    """Type-chart multiplier of `mtype` against a list of 1-2 defender types."""
    from poke_env.data import GenData

    ts = [t for t in types if t is not None]
    if not ts:
        return 1.0
    chart = GenData.from_gen(gen).type_chart
    return float(mtype.damage_multiplier(ts[0], ts[1] if len(ts) > 1 else None, type_chart=chart))


def tera_stab(mtype: PokemonType | None, mon: Pokemon, tera_type: PokemonType | None = None) -> float | None:
    """STAB multiplier of `mtype` if `mon` terastallized now (original types keep STAB;
    matching the Tera type too gives 2.0). None when the Tera type is unknown."""
    tt = tera_type if tera_type is not None else mon.tera_type
    if tt is None or mtype is None:
        return None
    base = [t for t in mon.types if t is not None]
    if mtype == tt:
        return 2.0 if tt in base else 1.5
    return 1.5 if mtype in base else 1.0


def stab_threat(attacker: Pokemon, defender: Pokemon, attacker_own: bool) -> tuple[float, float]:
    """Guess at the worst physical / special hit from an 80 BP STAB move (used when the
    attacker's moves aren't known yet)."""
    best = {MoveCategory.PHYSICAL: 0.0, MoveCategory.SPECIAL: 0.0}
    for t in attacker.types:
        if t is None:
            continue
        for cat in best:
            best[cat] = max(best[cat], raw_damage_frac(80, t, cat, attacker, defender, attacker_own))
    return best[MoveCategory.PHYSICAL], best[MoveCategory.SPECIAL]
