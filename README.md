# pokebot [OUTDATED]

RL agent for Pokémon Showdown singles (default: `gen9randombattle`), built on
[poke-env](https://github.com/hsahovic/poke-env) 0.16.

## Setup

```bash
pip install -r requirements.txt

# local Showdown server (needs Node 18+)
git clone --depth 1 https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown && npm install && cp config/config-example.js config/config.js
node pokemon-showdown start --no-security      # listens on :8000
```

## Run

```bash
python train_ppo.py --steps 200000 --opponent random      # warm up
python train_ppo.py --steps 500000 --opponent maxpower --resume checkpoints/latest.pt
python train_ppo.py --steps 1000000 --opponent heuristic --resume checkpoints/latest.pt
python evaluate.py --ckpt checkpoints/latest.pt --n 100
```

## Layout

- `pokebot/encoder.py` – battle -> 611-dim float vector (own team, seen opp team, own moves, boosts, field)
- `pokebot/env.py` – `PokemonEnv` (obs + shaped reward) and `make_env(opponent)`
- `pokebot/model.py` – masked-action MLP actor-critic
- `train_ppo.py` – PPO + GAE, single env, curriculum via `--opponent`
- `evaluate.py` – greedy policy vs random / max-power / heuristic bots

## Known limits of v0 (next steps)

- ~85 steps/s on one env: Showdown is the bottleneck. Run several servers/envs in parallel processes.
- Feed-forward policy has no memory of what the opponent has revealed. Add a GRU or history window.
- Encoder uses one-hots/base stats instead of learned species/move/item/ability embeddings.
- Shaping reward (HP/faint) should be annealed to pure win/loss.
- Next: self-play league with past checkpoints, then pretraining on Showdown replays.
