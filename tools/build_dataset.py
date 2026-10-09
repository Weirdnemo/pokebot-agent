"""Rebuild a decision dataset with any observation encoder from the raw room messages
saved by tools/collect_foulplay.py (data/<name>.npz + data/<name>.frames.jsonl.gz).

    python tools/build_dataset.py --in data/fp_x.npz --encoder v3 --out data/fp_x_v3.npz

Each battle's messages are replayed through poke-env's own Battle object exactly as the live
player processed them, and the new encoder is applied at the recorded decision points. The
legal-action mask recomputed during replay is compared with the stored one, which is the check
that the replay really reproduces the live state ("mask match" below). Needs only poke-env and
numpy (no Foul Play, no server).
"""
from __future__ import annotations

import argparse
import ast
import gzip
import json
import logging
import sys
from pathlib import Path

import numpy as np
import orjson

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pokebot.encoder import OBS_DIM, OBS_DIM_V3, encoder_for_dim  # noqa: E402


def replay_battle(frames, username, decisions, encode, get_mask):
    """decisions: list of (row index, frame_idx). Returns {row: (obs, mask)}."""
    from poke_env.battle import Battle
    from poke_env.player import Player

    tag = frames[0].split("\n")[0][1:]
    battle = Battle(battle_tag=tag, username=username, logger=logging.getLogger("replay"), gen=9)
    want = sorted(decisions, key=lambda d: d[1])
    out, k = {}, 0
    for n_done in range(len(frames) + 1):
        while k < len(want) and want[k][1] == n_done:
            out[want[k][0]] = (encode(battle), np.array(get_mask(battle), dtype=np.int8))
            k += 1
        if n_done == len(frames) or k >= len(want):
            break
        for sm in [m.split("|") for m in frames[n_done].split("\n")][1:]:
            if len(sm) < 2:
                continue
            kind = sm[1]
            if kind == "":
                battle.parse_message(sm)
            elif kind in Player.MESSAGES_TO_IGNORE or kind in ("error", "bigerror", "showteam"):
                continue
            elif kind == "request":
                if sm[2]:
                    battle.parse_request(orjson.loads(sm[2]))
            elif kind in ("win", "tie"):
                break
            else:
                battle.parse_message(sm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="the .npz written by collect_foulplay.py")
    ap.add_argument("--encoder", choices=["v2", "v3"], default="v3")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from pokebot.env import PokemonEnv

    inp = Path(args.inp)
    d = dict(np.load(inp, allow_pickle=False))
    meta = ast.literal_eval(str(d["meta"]))
    frames_file = inp.with_suffix(".frames.jsonl.gz")
    encode = encoder_for_dim(OBS_DIM_V3 if args.encoder == "v3" else OBS_DIM)

    by_tag: dict[str, list] = {}
    for i, (b, fi) in enumerate(zip(d["battle"], d["frame_idx"])):
        by_tag.setdefault(str(d["tags"][b]), []).append((i, int(fi)))

    n = len(d["obs"])
    new_obs = np.zeros((n, OBS_DIM_V3 if args.encoder == "v3" else OBS_DIM), np.float32)
    done = np.zeros(n, bool)
    mask_ok = obs_ok = 0
    nb = 0
    with gzip.open(frames_file, "rt") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["tag"] not in by_tag:
                continue
            res = replay_battle(rec["frames"], rec["user"], by_tag[rec["tag"]], encode, PokemonEnv.get_action_mask)
            for i, (obs, mask) in res.items():
                new_obs[i], done[i] = obs, True
                mask_ok += int(np.array_equal(mask, d["mask"][i]))
            nb += 1
            if nb % 200 == 0:
                print(f"  replayed {nb} battles", flush=True)
    # observation equality against the stored observations (same encoder only)
    if meta.get("encoder", "v2") == args.encoder:
        obs_ok = int(np.all(np.isclose(new_obs[done], d["obs"][done], atol=1e-6), axis=1).sum())
    keep = done
    print(f"replayed {nb} battles, {int(keep.sum())}/{n} decisions found; "
          f"mask match {mask_ok}/{int(keep.sum())} ({mask_ok / max(int(keep.sum()), 1):.4f})"
          + (f"; obs match {obs_ok}/{int(keep.sum())} ({obs_ok / max(int(keep.sum()), 1):.4f})"
             if meta.get("encoder", "v2") == args.encoder else ""))
    out = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n else v) for k, v in d.items()}
    out["obs"] = new_obs[keep]
    meta["encoder"] = args.encoder
    meta["rebuilt_from"] = str(inp)
    out["meta"] = np.array(str(meta))
    np.savez_compressed(args.out, **out)
    print(f"saved {args.out} with obs dim {out['obs'].shape[1]}")


if __name__ == "__main__":
    main()
