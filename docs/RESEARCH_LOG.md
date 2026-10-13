# Research log: Pokémon battle agent (poke-env / Pokémon Showdown)

Living lab notebook, written to support a possible paper. Rules for this file:

- Append-only for experiments (E-numbers never change). Corrections go in section 7, not by silent edits.
- Every number carries its sample size and who ran it: **[user]** = ran on the author's laptop, **[sandbox]** = ran by Claude in a cloud container.
- Win rates are wins/battles. "±" is the 95% normal-approximation half-width; at these sample sizes treat anything inside it as noise.

Last updated: 2026-10-13. Authors: Nimesh Chauhan, with Claude (Anthropic) as coding assistant.

---

## 0. One-paragraph status

A small MLP policy over hand-engineered features (811 inputs) was trained for Gen 9 random battles. Behaviour cloning (BC) from the built-in `SimpleHeuristicsPlayer` reached teacher-level strength (0.99 / 0.90 / 0.45 vs random / max-power / heuristic bots) in about one hour of wall-clock time, whereas PPO from scratch reached 0.92 / 0.70 / 0.19 after ~1M steps. PPO fine-tuning from the BC checkpoint did not improve on it: without a regulariser it degraded (0.91 / 0.76 / 0.47), with a KL anchor to the BC policy it held (0.97 / 0.83 / 0.44 at ~1.1M cumulative steps) but never beat BC. Plan after E7 was to stop scaling PPO and use the public strong agents as teachers. That was tried (E8-E13): distilling Foul Play's search policy into the MLP, with 12x more data (E11), Tera features and DAgger-style data (E13), did not raise the win rate above the BC clone (about 0.45 vs the heuristic bot throughout). Current plan (decided 2026-10-13): give the network a memory of what it has seen of the opponent, then fine-tune with anchored RL.

---

## 1. Goal and scope

- Goal: a "strong contender" Pokémon Showdown agent that matches the capabilities of top public agents while training and running on one laptop. Not state of the art; RL should be part of the solution, not the only part.
- Format used so far: `gen9randombattle` (teams are assigned, so no team building), singles, local Showdown server.
- Opponents used so far (all from poke-env 0.16.1): `RandomPlayer`, `MaxBasePowerPlayer` ("maxpower"), `SimpleHeuristicsPlayer` ("heuristic").

## 2. Environment and versions

| Item | Author's laptop [user] | Claude's sandbox [sandbox] |
|---|---|---|
| Machine | Lenovo IdeaPad Gaming 3 15ARH7, Fedora 44 | cloud container, Linux 6.18 |
| CPU | Ryzen 5 6600H, 12 threads | not recorded |
| GPU | RTX 3050 Mobile (training here uses CPU unless CUDA is present) | none used |
| RAM | 14.8 GiB | not recorded |
| Python | 3.10 (conda env `poke`) | 3.13 |
| poke-env | 0.16.1 | 0.16.1 |
| torch / gymnasium | not recorded | 2.14.1 / 1.3.0 |
| Showdown | cloned from smogon/pokemon-showdown master (commit not recorded; please run `git rev-parse HEAD` and add it) | commit `9fb3a5b99f1a0bea17f495c5cc1bfe04fdd19c3e` (2026-10-03) |

Server started with `node pokemon-showdown start --no-security`.
Throughput: ~140-240 env steps/s on one environment [user logs: 213, 239, 189, 177 sps in different runs].

## 3. Code (repo `pokebot/`)

| File | Purpose |
|---|---|
| `pokebot/encoder.py` | Battle -> fixed float vector. v0 = 611 dims, v2 = 811 dims |
| `pokebot/damage.py` | Cheap expected-damage and speed estimates used as features |
| `pokebot/env.py` | `PokemonEnv` (SinglesEnv subclass), shaped reward, `make_env(opponent)`; `strict=False` default |
| `pokebot/model.py` | Masked-action MLP actor-critic (512-512, LayerNorm+ReLU, separate heads) |
| `train_ppo.py` | PPO + GAE; `--opponent random/maxpower/heuristic/mixed`; `--kl-coef --anchor --save-every` |
| `pretrain_bc.py` | Collect teacher decisions + reward-to-go, behaviour-clone policy and value head |
| `evaluate.py` | Greedy (argmax) policy vs the three baselines |
| `bench_external.py` | Play our bot or a checkpoint against an external bot (Foul Play etc.) |

### Observation (v0, 611 dims)
Own 6 Pokémon and the opponent's seen Pokémon (HP fraction, fainted, active flag, status one-hot, type multi-hot, base stats/255, tera flag, known flag); our 4 active moves (base power, accuracy, PP fraction, type, category, priority, type effectiveness vs opponent active); boosts for both actives; weather; hazards/screens for both sides; tera availability, turn/50, alive counts.

### Observation (v2, 811 dims = v0 + 200)
Adds: per-move expected damage as a fraction of opponent max HP and of opponent current HP, STAB flag; the opponent's revealed moves (shape + estimated damage to our active); an 80-BP STAB threat guess from the opponent to each of our six Pokémon (physical/special); each bench Pokémon's moves vs the opponent's active (damage, effectiveness, best damage); speed ratio (log2, clipped) and a faster flag. Damage uses the standard formula with estimated stats (31 IVs, 85 EVs, neutral nature) for unseen stats.

### Reward
`reward_computing_helper(fainted_value=0.15, hp_value=0.05, victory_value=1.0)`: change in a state value made of HP fractions, faints and the +/-1 win term (hand-shaped; suspected of misaligning optimisation, see H2).

### PPO hyperparameters (unless stated)
rollout 2048, 4 epochs, minibatch 256, gamma 0.99, GAE lambda 0.95, clip 0.2, value coef 0.5, entropy coef 0.01, grad-clip 0.5, Adam, lr 3e-4 (1e-4 when fine-tuning from BC). Action masking via logits set to -1e9 for illegal actions.

---

## 4. Experiments

Evaluation protocol for all "eval" rows: `evaluate.py`, greedy policy, `gen9randombattle`, n battles per opponent, local server. Training-log win rates (last 100 battles, sampled actions) are noted separately because they run lower than greedy evaluation.

### E1: PPO from scratch vs random, v0 features [user]
- Run: ~61k steps against `random`.
- Training log at 61,440 steps: win rate (last 100) 0.90.
- Eval (n=100 each): random 0.93 (±0.05), maxpower 0.44 (±0.10), heuristic 0.09 (±0.06).
- Reading: learns basic move selection; far from the stronger bots.

### E2: PPO vs maxpower, v0 features [user]
- Run: ~500k further steps against `maxpower` (resumed).
- Training log at 485k-502k steps: win rate vs maxpower 0.37-0.53 (last-100 values 0.50, 0.48, 0.47, 0.49, 0.37, 0.41, 0.53, 0.51, 0.50).
- No greedy eval taken. Plateau at ~0.5.
- Reading: no progress from more steps with v0 features. Triggered the v2 feature work.

### E3: v2 features, mixed opponents (20% random / 40% maxpower / 40% heuristic) [user]
- Instructions given: 100k warm-up steps vs random, then ~1M mixed steps (the warm-up was part of the recommended command sequence; whether it was run exactly so is **unconfirmed**).
- Training log at ~291-295k steps: random 0.99, maxpower 0.47-0.53, heuristic 0.16-0.18.
- Training log at 1,001,472 steps: overall 0.55 [random 0.91, maxpower 0.64, heuristic 0.19].
- Eval at ~1.0M steps (n=100): random 0.92 (±0.05), maxpower 0.70 (±0.09), heuristic 0.19 (±0.08).
- Confound: E1/E2 (v0) vs E3 (v2) differ in features, step count and opponent mix at the same time, so the maxpower gain (0.44 -> 0.70) is not attributable to features alone. A clean ablation has not been run (see H5).

### E4: Teacher measurement, `SimpleHeuristicsPlayer` driving our env [user]
- 5000 battles with opponent mix 20/40/40 (random/maxpower/heuristic), 110,900 decision points collected.
- Teacher win rates: random 0.98 (n=1001), maxpower 0.89 (n=2025), heuristic 0.49 (n=1974). Heuristic-vs-heuristic is ~0.5 by symmetry, which is a sanity check.
- Collection time 561 s.
- [sandbox] 100-battle pilot: 1.00 (n=13) / 0.85 (n=41) / 0.54 (n=46). Consistent but tiny.

### E5: Behaviour cloning from the heuristic teacher [user]
- Data: E4's 110,900 decisions (only states with more than one legal action). Targets: teacher action (cross-entropy) and discounted reward-to-go (value MSE, coef 0.5). gamma 0.99.
- Training: 10 epochs, lr 1e-3, batch 512, 5% random hold-out split.
- Hold-out action accuracy per epoch: 0.796, 0.826, 0.846, 0.861, 0.852, 0.857, 0.870, 0.870, 0.868, 0.872. Value MSE 0.505 -> 0.073. Training loss kept falling (1.171 -> 0.112) while hold-out accuracy plateaued from epoch ~4-7 (mild overfitting).
- Eval (n=100): random 0.99 (±0.02), maxpower 0.90 (±0.06), heuristic 0.45 (±0.10).
- Reading: matches the teacher within noise. This is the strongest agent so far.
- Pre-registered prediction (made before the eval): 0.70-0.80 vs maxpower and 0.30-0.40 vs heuristic. The agent exceeded it. Recorded to keep predictions honest.

### E6: PPO fine-tuning from BC, no regulariser [user]
- Setup: mixed opponents, lr 1e-4, entropy coef 0.01, no KL term, resumed from `bc.pt`.
- Training log: 0.72 overall at 2,048 steps (random 1.00, maxpower 0.86, heuristic 0.38; ~30 games each, so noisy); at 415k-424k steps: random 0.94-0.97, maxpower 0.71-0.75, heuristic 0.33-0.37.
- Eval of `checkpoints_bc/latest.pt` at ~420k steps (n=200): random 0.91 (±0.04), maxpower 0.76 (±0.06), heuristic 0.47 (±0.07).
- Reading: PPO eroded performance against the easy opponents (maxpower 0.90 -> 0.76) and did not improve heuristic. Suspected causes: the entropy bonus working against a sharp cloned policy and/or the shaped reward (H2).

### E7: PPO fine-tuning from BC with a KL anchor [user]
- Setup: mixed opponents, lr 1e-4, entropy coef 0.001, KL(policy || BC policy) penalty coefficient 0.2 (masked), `--save-every 100000`.
- KL value in logs stayed ~0.05 (0.048-0.056 at 37k-47k steps).
- Run 1 crashed at ~112k steps: a `ValueError: 'outrage' is not in list` raised inside poke-env's `order_to_action` when the *scripted opponent* chose a locked-in move. Fix: `make_env` now defaults `strict=False` (falls back to a random legal move). Frequency: 1 crash in ~4,100 battles.
- Eval `step_100352.pt` (n=200): random 0.98 (±0.02), maxpower 0.86 (±0.05), heuristic 0.47 (±0.07).
- Run 2 (`checkpoints_kl2`): resumed from run 1's latest with `--anchor checkpoints/bc.pt`; step counter restarted, so run-2 step N is about N + 112k cumulative.
  - `step_401408.pt` (~513k cumulative, n=200): 0.98 (±0.02) / 0.84 (±0.05) / 0.45 (±0.07).
  - `step_1001472.pt` (~1.11M cumulative, n=200): 0.97 (±0.02) / 0.83 (±0.05) / 0.44 (±0.07).
- Reading: the anchor prevents the E6 degradation but gives no gain; all E7 checkpoints are within noise of each other and of BC (0.99 / 0.90 / 0.45), except possibly a small drop vs maxpower (0.90 -> 0.83, borderline at these n).

### E8: Step 0, external reference agents (Foul Play, Metamon) [sandbox]
Goal: measure the real strength yardstick. **Outcome: partially blocked, benchmark must be run on the author's machine.**

- Foul Play (`pmariglia/foul-play`, commit `8f541f80f91f960603673796559f796b10ccf0fb`, GPL-3.0): installs in a fresh venv in ~51 s (Python 3.13, Rust 1.97, `poke-engine==0.0.48` built with the `terastallization` feature, i.e. the Gen 9 engine). Starts and waits for challenges on the local server.
  - Needed a local-test patch (not upstream): its login always calls `play.pokemonshowdown.com`, which the sandbox network blocks. Patch skips it when `FP_LOCAL_NO_AUTH=1`; the local `--no-security` server accepts an empty assertion. Saved as `tools/foul_play_local_noauth.patch` (apply with `git apply` inside the foul-play clone). **Correction (2026-10-07): it is required for local-server runs on any machine, not only in the sandbox; see the login finding below.** (An earlier version of this log said it was not needed where the official login service is reachable. That was wrong.) The sandbox clone itself was restored to pristine upstream and nothing was pushed to the upstream repository.
  - For `gen9randombattle` it downloads set data from `pkmn.github.io/randbats/...`, also blocked in the sandbox, so no battle was played. **No Foul Play strength number exists yet.**
- Metamon (`UT-Austin-RPL/metamon`, v1.6.0, commit `0a00a759c9a4382a2877088d828302ec294a05a5`, MIT): needs its own poke-env fork and `amago` v3.4.0, `gymnasium<=0.29.1`, `torch>=2.6`, so it must live in a separate environment. Models and datasets are on Hugging Face, which the sandbox cannot reach, so nothing was downloaded or run.
  - Supported formats: Gen 1-4 OU/NU/UU/Ubers and **Gen 9 OU only. No random battles.** This forces a format decision (section 6).
  - Smallest model suited to a laptop: `Minikazam` (4.7M-parameter RNN). Larger: `Abra` (57M, Gen 9 OU), `Kakuna` (142M, best public).
- `bench_external.py` was tested against a stand-in random bot in a separate process: our heuristic bot won 4/4 in `--mode accept` (our bot waits, the external bot challenges). `--mode challenge` hung in two attempts, cause unknown; use accept mode.
- **Handshake failure on the author's machine (2026-10-07) and diagnosis [sandbox].** Symptom: our bot waiting, Foul Play in `challenge_user` mode prints "Challenging OurBot..." and nothing else happens. Reproduced once in the sandbox with a checkpoint bot started only 10 s before Foul Play. Challenges sent before our bot has logged in are silently dropped by Showdown and Foul Play never retries, so it waits forever. With our bot confirmed logged in first (heuristic bot after 8 s; checkpoint bot after 30 s and again after an explicit ready signal), Foul Play logged "Initialized battle-gen9randombattle-577 against: OurBot", i.e. the handshake works. The first-failure startup delay in the sandbox (container had just restarted, cold file cache, `import torch`) is a plausible but unproven cause; on the author's machine the cause is not yet confirmed. Fix: `bench_external.py` now prints a `[ready]` line once logged in; start the external bot only after seeing it.
- **Login finding (2026-10-07), the leading explanation for the author's stall.** After the `[ready]` fix the stall persisted on the author's machine: bench printed `[ready]`, Foul Play printed "Successfully logged in" then "Challenging OurBot..." and no battle appeared in the local Showdown lobby. [sandbox] test: a local `--no-security` server accepts `/trn <name>,0,` with an *empty* assertion (`|updateuser| <name>|1|...`) but rejects an invalid one (`|nametaken|<name>|Your authentication token was invalid.`), leaving the connection an unnamed guest. Foul Play always requests its guest assertion from the official login service using the local server's challstr; the official service did not issue that challstr, so the assertion is invalid (the exact response could not be observed because the sandbox cannot reach the site). Foul Play does not check the result of `/trn` and still logs "Successfully logged in", so it likely plays as "Guest N". Its challenge then comes from the guest name; our bot accepts only from "FoulPlayBot", hence silence. Fix: apply `tools/foul_play_local_noauth.patch` and run with `FP_LOCAL_NO_AUTH=1`. Status: **confirmed (2026-10-07).** With the patch applied and `FP_LOCAL_NO_AUTH=1`, Foul Play logged in as FoulPlayBot and battles ran on the author's machine.
- **First Foul Play numbers (author's machine, gen9randombattle, local server, accept mode).** SimpleHeuristicsPlayer vs Foul Play: n=5: 0/5; n=100: **6/100 = 0.06 (95% Wilson CI 0.03-0.12)**. Foul Play search settings, recovered afterwards from the process command line (2026-10-07): `python run.py --websocket-uri ws://localhost:8000/showdown/websocket --ps-username FoulPlayBot --pokemon-format gen9randombattle --bot-mode challenge_user --user-to-challenge OurBot --run-count 100`, i.e. **no search flags, so the defaults** (in the upstream commit we cloned, `8f541f8`: `--search-time-ms 100`, `--search-parallelism 1`; defaults on the author's copy not separately confirmed). Debug output from a game shows about 47-50k MCTS iterations per sampled world and two sampled opponent worlds per decision. **Record the flags for every later run**, since they set Foul Play's strength. Reading: the scripted teacher that BC cloned (parity ~0.45 vs our heuristic eval, 0.99 vs random) is far weaker than a search agent here. Our BC agent has not yet been run against Foul Play; given BC ~ teacher parity it is expected to land near 0.06, which is the baseline gap that H1 (search-teacher DAgger) must close.
- **BC checkpoint vs Foul Play (author's machine, same setup, n=100): `checkpoints/bc.pt` 7/100 = 0.07 (95% CI 0.03-0.14).** This matches the prediction and is statistically indistinguishable from the teacher (0.06, CI 0.03-0.12). Same Foul Play command line as the heuristic run (default search settings). So against a search agent, BC parity with the teacher means about 6-7% win rate; behaviour cloning the heuristic teacher inherits its ceiling.
### E9: Foul Play decision recorder (`tools/collect_foulplay.py`) [sandbox-tested, not yet run on the author's machine]
Goal: get (our observation, Foul Play's search policy) pairs for distillation (H1 step 1).

- Design. One process, one account. A poke-env `Player` owns the websocket and keeps its own `Battle` current; each raw room message is forwarded verbatim to Foul Play's own `pokemon_battle` loop **after** poke-env has processed it, through a queue-based stand-in for Foul Play's websocket client. Foul Play's sends are intercepted; when it picks a move we record `encode_battle(battle)` (811-d), the 26-action legal mask, Foul Play's choice mapped to our action space, and its full search policy (visit shares over all options, summed over the sampled opponent worlds, weighted by sample chance, then mapped to our action space and normalised), then forward the choice to the server. Two Foul Play functions are wrapped at run time (`fp.modes.base.find_best_move`, `fp.search.main.select_move_from_mcts_results`) only to read their results; the choice itself is still made by Foul Play's own code. Opponent = a built-in poke-env bot in the same process (heuristic by default). No second terminal, no login patch needed (Foul Play never logs in). Foul Play search flags (`--search-time-ms`, `--search-parallelism`) are passed through and stored in the output's `meta`.
- Output `.npz`: `obs, mask, act, policy, n_legal, turn, battle, outcome (+1/-1 from Foul Play's side), meta`.
- Sandbox test (10 battles vs heuristic, Foul Play at defaults 100 ms x1): 225 decisions; chosen action legal in 100%; policy sums to 1 and puts no mass on illegal actions; **0 unmapped decisions**; the sampled choice equals the argmax of the policy in 89% of decisions (Foul Play samples among options within 75% of the best, so < 100% is expected); 0.9% of decisions had a single legal action. About 6 s per battle. **Caveat: the sandbox could not download Foul Play's random-battle set data, so this test used a synthetic stand-in built from Showdown's own `sets.json` (all items "leftovers", flat counts). That exercises the plumbing and the real search but not realistic set inference; Foul Play's strength and its decisions in that test are not representative.** The stand-in file was deleted afterwards; on the author's machine Foul Play downloads the real file on first use.
- **First real run (author's machine, real set data, 20 battles vs heuristic, Foul Play defaults): 451 decisions, 1 unmapped (0.2%), 0 Foul Play errors, 114 s (~5.7 s/battle), Foul Play won 20/20.** Foul Play printed some harmless-looking warnings of its own ("steel/ghost is not a known move", one volatile-status mismatch); not investigated. Consequence: **against the heuristic opponent Foul Play almost never loses, so the outcome (value) label is almost constant (+1).** The policy distillation is unaffected, but a value head trained on this data learns little; value learning needs games Foul Play can lose (stronger or mixed opponents, or self-play).
- Open question: how often the poke-env and Foul Play views of the state disagree (not measured; the `unmapped` counter is only a partial check).
- Licence: imports Foul Play (GPL-3.0) as a library and patches two functions at run time; no Foul Play source is copied here. Check this before releasing code.

- **500-battle collection (author's machine, 2026-10-08, Foul Play defaults 100 ms x1, opponent heuristic):** 11,146 decisions, 13 unmapped (0.12%), 0 Foul Play errors, 2,877 s (5.8 s/battle), Foul Play won 473/500 (94.6%), consistent with the 6/100 heuristic result against it. Saved as `data/fp_500.npz`.

### E10: distilling Foul Play's search policy into our MLP (`pretrain_fp.py`), first results
Data: `data/fp_500.npz`; split by battle: 450 battles / 10,005 decisions train, 50 battles / 1,141 decisions validation; 811-d observation, 26 actions, MLP 512-512 (same as before). Teacher entropy on validation 1.44 nats.

| run | best epoch val `agree_argmax` | last-epoch val `agree_argmax` | val policy CE: first -> last epoch | note |
|---|---|---|---|---|
| soft targets, from scratch | 0.33 (epoch 1) | 0.32 | 1.97 -> 2.10 | train CE 2.12 -> 1.50 |
| soft targets, init from `bc.pt` | **0.41 (epoch 1)** | 0.34 | 1.96 -> 2.10 | |
| hard labels (picked move only), from scratch | 0.29 (epoch 1) | 0.28 | 1.75 -> 3.41 | train CE 1.94 -> 0.33, student entropy 0.72 vs teacher 1.44 |

(`agree_argmax` = how often our top choice equals Foul Play's top search choice; chance among legal actions is about 0.12.)

- Reading (stated carefully): **all runs overfit after about one epoch** (train loss keeps falling, validation loss rises), and the best agreement is only 0.33-0.41. The hard-label run overfits worst, which is weak evidence that soft targets help, but its epoch-1 validation CE (1.75) is not comparable with the soft-target runs' (the loss functions differ), so this table does not show it directly. Starting from `bc.pt` was better after one epoch (0.41 vs 0.33); with one seed and 1,141 validation decisions the standard error on each agreement figure is about 0.015, so that gap is probably real but untested across seeds.
- **Play strength of the distilled checkpoints (author's machine, `evaluate.py`, n=200 each, greedy argmax; same opponents as E3-E7):**

| checkpoint | vs random | vs maxpower | vs heuristic |
|---|---|---|---|
| `checkpoints/bc.pt` (BC from heuristic, earlier result) | 0.99 | 0.90 | 0.45 |
| `checkpoints_fp_init/fp_best.pt` (Foul Play soft targets, early-stopped) | 0.99 | 0.82 | 0.46 |
| `checkpoints_fp_init/fp.pt` (same run, last epoch, overfit) | 0.96 | 0.74 | 0.28 |

  Reading: at 10k decisions the distilled network plays at about BC level against the heuristic (0.46 vs 0.45) and possibly a little worse against maxpower (0.82 vs 0.90; the standard error of each is about 0.02-0.03, so the difference is borderline). **Overfitting directly costs playing strength** (0.28 vs the heuristic for the last epoch vs 0.46 for the early-stopped one), so early stopping on validation loss is not optional. So far the distillation has not beaten plain behaviour cloning of the heuristic; it has not been tested against Foul Play itself yet.
### E11: scaling the distillation data about 12x (author's machine, 2026-10-08)
Data: `fp_500` + six parallel collections (about 940 battles each at the largest; 5 parallel opponent mixes: heuristic x3 incl. `fp_500`, maxpower x1, random x3 in total, the exact per-file counts are in the loader output), all Foul Play defaults (100 ms x1), collected with 6 collectors in parallel at 50-60% total CPU (a possible small weakening of the teacher through fewer search iterations per 100 ms; not measured). One collector (`fp_b`) hung on a Foul Play `ZeroDivisionError` in battle 941; its first 940 battles (saved every 10) are used. Two training sets, both initialised from `checkpoints/bc.pt`, split by battle:

| set | decisions / battles (train+val) | val agree_argmax at epoch 0 (= `bc.pt` itself) | best val agree_argmax after training | val policy CE: epoch 0 -> best | evaluate.py n=200 (random / maxpower / heuristic) |
|---|---|---|---|---|---|
| `checkpoints_fp_big` (all files) | 130,420 / 6,440 | **0.437** | 0.447 | 9.01 -> 1.91 | 1.00 / 0.80 / 0.49 |
| `checkpoints_fp_noRandom` (no random-opponent files) | 72,720 / 3,440 | 0.409 | 0.420 | 8.99 -> 1.90 | 0.98 / 0.85 / 0.41 |
| `bc.pt` for reference (earlier) | n/a | n/a | n/a | n/a | 0.99 / 0.90 / 0.45 |

Reference numbers printed by the script: chance among legal actions about 0.11-0.12; a perfect copy of Foul Play's top choice would match its sampled pick 0.86-0.87 of the time; teacher entropy 1.48-1.49 nats (so the cross-entropy cannot go below about 1.49; we end at about 1.90, i.e. about 0.4 nats above it).

- **Reading.** (1) The old BC network, trained only to copy the heuristic bot, already agrees with Foul Play's top choice 41-44% of the time. Training on 12x more Foul Play data moves that to 42-45%, i.e. **about +1 point**. The big drop in cross-entropy (9.0 -> 1.9) is mostly calibration: `bc.pt` is extremely confident (student entropy 0.20 vs teacher 1.49), so every miss is heavily penalised. (2) Play strength did not measurably change: 0.49 / 0.41 / 0.45 against the heuristic are within the roughly 0.035 standard error of n=200 of each other; against maxpower both new checkpoints are at or below BC (0.80, 0.85 vs 0.90), borderline. (3) Early stopping triggered after 8-9 epochs in both runs. (4) Dropping the random-opponent files lowered agreement (0.42 vs 0.45) on a different validation set, so the two agreement numbers are not directly comparable; the play results do not favour either set.
- **So more data of this kind did not help, and the earlier plan "scale data, then train" is not enough by itself.** What this leaves open: whether the limit is the observation (hidden-information beliefs and search lookahead are not visible to the MLP), the model, or the offline-imitation setup (our network never sees states produced by its own mistakes; DAgger, H1). `analyze_fp.py` was written to split these by action type, search confidence, turn and number of legal actions.
- **Error analysis (`analyze_fp.py` on `checkpoints_fp_big/fp_best.pt`, 12,723 validation decisions with >1 legal action, overall agreement 0.445):**
  - By Foul Play's top-choice type: *switch* 19.2% of decisions, agreement 0.159 (our net plays a move instead in 54.6% of those; Foul Play plays a move and we switch in only 1.1%, so we **under-switch**); *move* 77.1%, agreement 0.537; *terastallize-move* 3.7%, agreement **0.004** (we predict a Tera move as top choice in about 0.2% of decisions, i.e. **the network practically never terastallizes**).
  - By how peaked Foul Play's search was (top option's share): <0.4: 41.9% of decisions, agreement 0.348; 0.4-0.6: 30.2%, 0.462; 0.6-0.8: 14.3%, 0.558; >0.8: 13.6%, 0.589. So (i) in 42% of decisions the search itself is spread out, which caps top-1 agreement for any student, so `agree_argmax` is a harsh metric; (ii) but even where the search is decisive (>0.8) we miss 41%.
  - By turn: 0.413 (turns 1-4), 0.437 (5-11), 0.462 (12-21), 0.464 (22+). Agreement rises only slightly as more of the opponent is revealed. This is weak evidence against the idea that missing opponent-set beliefs are the dominant limit (confounded; not a test).
  - By number of legal actions: 0.43 with 8+ legal actions (77% of decisions), 0.51 with 5-7, 0.42 with 3-4.
  - Code check (grep of `pokebot/encoder.py`): the observation has the "can terastallize" flag and an "is terastallized" flag per mon, but **no Tera type** of our own or the opponent's Pokemon, and no tera-adjusted damage. A concrete, fixable feature gap, but it covers at most about 3.7 points of agreement.
- Not yet measured: how the `fp_big` checkpoint plays against Foul Play itself (expected near the 0.06-0.07 of the BC agent).

- Open: (a) what the ceiling is: a perfect copy of the argmax would match Foul Play's sampled pick in ~89% of decisions in the sandbox test (sampled among options within 75% of the best), and the training script now prints this reference for the real data; two independent Foul Play searches on the same state may agree with each other much less than that, which we have not measured; (b) whether more data (10k decisions is small for an 811-d input) or better features (set-belief features, H3) close the gap; (c) whether the checkpoints actually play better, not yet benchmarked.

- Sandbox-only artefact: a later Foul Play `KeyError: 'battle\n'` appeared after I had killed earlier test battles mid-game on the same server (a reused room id); not seen elsewhere, ignored.

### E12: literature check, then encoder v3 + replayable recorder + DAgger mode (2026-10-09) [sandbox-tested]
**Literature check (what was actually read this session; a first delegated pass hit a search quota and covered only part, the rest was fetched directly).**
- Metamon, arXiv 2504.04395 (full text fetched): no search; transformer policies with long context (15M / 50M / 200M parameters) that infer the opponent's hidden team from the history of the battle (only the opponent's active Pokemon is observed); observation = about 87 words of text (species, items, abilities, moves) + 48 numeric features. Offline RL beat pure behaviour cloning; synthetic self-play data (2M-5M trajectories) gave the largest gains; self-play fine-tuning helped against itself but gave inconsistent gains against humans. Gen 1-4 OU. Reported SynRL-V2 in roughly the top decile of the human ladder (Gen 1 OU GXE 79.9% over 613 games). Against Foul Play the paper reports wins in Gens 1-2 and mixed results in Gens 3-4 (300 battles per gen; exact numbers not legible in the fetched text). Compute (GPUs, hours) is not stated in the text we got.
- PokeAgent Challenge, arXiv 2603.15563 (read by the delegated pass, not independently re-read): Foul Play = root-parallel MCTS with decoupled UCT on poke-engine; reported over 90% GXE in Gen 9 random battles (peak rating 2341) and over 80% in Gen 9 OU. Gen 1 OU winner built on Metamon with iterative offline RL; the Gen 9 OU runner-up (Team Q) used a 50M-parameter actor-critic trained against heuristic bots, then a fixed expert "coach" and its own past versions. No winning-team compute figures were reported.
- PokeChamp, arXiv 2503.04094: LLM-guided minimax (LLM proposes actions, models the opponent, scores leaves), no training; Gen 9 OU, 84% against the Abyssal bot with GPT-4o, estimated human-ladder Elo 1300-1500; about a third of its ladder games were lost on the clock; compute not reported. Needs an LLM per move, so not a laptop option.
- DAgger: Ross, Gordon, Bagnell 2011, arXiv 1011.0686 (found by search; the "error grows quadratically with horizon for plain cloning, linearly with DAgger" result is from my knowledge of that paper and was not re-read here).
- **Not found despite searching:** how Foul Play's strength scales with search time; documented failure modes of distilling search into a policy; belief-model ablations in Pokemon.
- Implications (my inference, not findings): (1) the strong learned agents rely on long-history sequence models and millions of trajectories; our 0.8M-parameter MLP with no history is three orders of magnitude below that scale, so the laptop shortcut is to use Foul Play as the labeller, not to reproduce Metamon; (2) every strong learned system used RL or self-play beyond imitation, which supports trying anchored RL after DAgger; (3) hidden-information inference through history is a likely missing ingredient (candidate: history features or a recurrent encoder).

**Code changes (all sandbox-tested; not yet run on the author's machine):**
- `pokebot/encoder.py`: encoder **v3** = v2 + 156 Tera features (own Tera type per mon, opponent active's Tera type once terastallized, any-terastallized flags, own damage if we terastallize now and the gain, the opponent's revealed-move damage against our Tera-type defence). Observation size 967 (v2: 811). Finding while testing: poke-env 0.16.1 does not read our mons' Tera types from the request (`side.pokemon[i].teraType`), so the type must come from `battle.last_request`; the first draft of the feature used `mon.tera_type` and silently saw nothing. `pokebot/env.py`, `evaluate.py`, `bench_external.py` and the loader `load_actor_critic` choose the encoder from a checkpoint's input size, so v2 checkpoints keep working. PPO training (`train_ppo.py`) is still v2 only.
- `tools/collect_foulplay.py`: stores the raw room messages (`<out>.frames.jsonl.gz`) and each decision's position in them; `--encoder v2|v3`; **DAgger mode** `--student ckpt --beta B` (with probability B Foul Play's action is played, otherwise our network's; Foul Play's search policy at that state is recorded either way; prints how often the student agrees with Foul Play on the states it visits); a Foul Play crash in one battle now forfeits that battle and continues (the outcome label of such a battle is set to 0).
- `tools/build_dataset.py`: rebuilds observations with any encoder from the saved messages. **Replay check (6 battles, sandbox): rebuilt masks and rebuilt observations equal the live ones in 128/128 (v2) and 117/117 (v3) decisions.** Rebuilding a v2-collected file with encoder v3 also worked (mask match 128/128).
- `pretrain_fp.py`: input size from the data, `--hidden`, warm start from a checkpoint with fewer inputs (the new input columns start at zero), `--patience` early stopping.
- DAgger smoke test (random-initialised student, beta 0.5, 6 battles): the played action equals the student's whenever it differs from Foul Play's; Foul Play's win rate falls (2/6) as expected with a random student.
- Caveats: sandbox tests used a synthetic stand-in for Foul Play's set data (see E9). In DAgger mode Foul Play's internal record of its own last move assumes its own action was played, which is not true when the student's action is played; the effect on its search at the next turn was not measured. Outcome labels in mixed games describe the mixed policy, not Foul Play.

---

### E13: v3 features + Foul Play data with 600 DAgger battles, trained and evaluated (author's machine, 2026-10-12/13) [user]
**Data (all with encoder v3, Foul Play 100 ms, 1 thread, `gen9randombattle`):** `v3_a` 600 battles / 13,456 decisions and `v3_b` 600 / 12,962 vs the heuristic bot; `v3_c` 600 / 11,282 vs max-power; `v3_d` 300 / 8,123 and `v3_e` 300 / 8,208, DAgger mode with the v2 student `checkpoints_fp_big/fp_best.pt`, beta = 0.5, vs the heuristic. Total 54,031 decisions, 2,400 battles. DAgger runs: student agreed with Foul Play's top choice on about 0.24-0.27 of the states it visited. Foul Play (as the mixed policy) won 154/300 in `v3_e`; those win rates describe the mixture, not Foul Play. Counts: `unmapped` 10 of 8,208 and `fp_errors` 1 in `v3_e`; no stalls after the watchdog (below).

**Training (`pretrain_fp.py`, init from `checkpoints_fp_big/fp_best.pt`, 22 new input columns zero-padded, hidden 512, early stopping):**
- Epoch 0 (before any update): val policy_ce 1.874, agreement with the search argmax 0.391, H_teacher 1.43, H_student 1.86.
- Epoch 6 (early stop): train loss 1.723, val policy_ce 1.902, agreement 0.385. The validation loss never fell meaningfully below its starting value and rose while the training loss fell (overfitting from the first epochs). The best-epoch validation loss was not printed in what was reported to me.
- Reference: random-legal agreement about 0.129; an exact copy of the argmax agrees with Foul Play's *sampled* pick 0.874 of the time.

**Error analysis (`analyze_fp.py`, 5,352 validation decisions with more than one legal action; overall agreement 0.399):**

| Foul Play's top choice | share | our agreement |
|---|---|---|
| switch | 23.3% | 0.191 |
| move | 73.5% | 0.482 |
| tera move | 3.2% | 0.000 |

- Confusion: when Foul Play switches, we predict a switch 55.2% of the time and a move 44.8%; when it plays a move or a Tera move we almost never predict a switch (1.8%, 2.4%) and predict a Tera move at most 0.1%.
- Derived (my arithmetic, not printed): picking the right switch target given that we predicted a switch is about 0.191 / 0.552 = 0.35.
- Agreement rises with how peaked the search was: 0.33 (top share under 0.4; 38% of decisions), 0.40, 0.48, 0.50 (above 0.8). It is flat across turns (0.39-0.41) and across the number of legal actions.

**Play strength (`evaluate.py --ckpt checkpoints_v3/fp_best.pt --n 200`):** 0.98 vs random, 0.81 vs max-power, **0.45 vs heuristic**. Earlier runs on the same evaluation: BC clone 0.99 / 0.90 / 0.45; E11 big v2 run 1.00 / 0.80 / 0.49; no-random v2 run 0.98 / 0.85 / 0.41. At n = 200 the 95% half-width is about 0.07, so every run is within noise of every other.

**Reading (inference):**
1. The Tera features did not change measured play. The agreement metric cannot show it either way: Tera is about 3% of decisions, so a model with a correct Tera rate would almost never rank a Tera action first. Only play can show whether the Tera timing improved, and play did not change.
2. The remaining visible gap is switching: the network senses a switch is coming but picks the wrong target in about two thirds of those cases. Choosing the target depends on beliefs about the opponent's hidden set, which the encoder does not carry.
3. More teacher data and on-policy relabelling did not help: validation loss overfits within a few epochs, and play strength stays at the level of the heuristic clone. By the decision rule stated beforehand (heuristic win rate about 0.46 means stop collecting), the plain distillation line stops here.
4. Not tested: the larger-hidden-size run (`--hidden 1024`), a DAgger round 2 with a v3 student, a direct benchmark of this checkpoint against Foul Play itself.

**Infrastructure issues found and fixed during this experiment (for reproducibility):**
- DAgger mode crashed Foul Play in about 44% of battles (83 of 190 in one run; `KeyError: 'switch<name>'` in `check_speed_ranges`). Cause: when our network's action replaced Foul Play's, Foul Play still recorded its own choice as `last_selected_move`; if it had chosen a switch and we played a move, its speed-inference code looked up the "move" `switch<name>`. Fix: the recorder now sets Foul Play's `last_selected_move` to the action actually played. Verified in the sandbox (30 battles with a random student overriding Foul Play almost every move: 0 crashes); on the author's machine `fp_errors` went to 0 in the first 300 battles and 1 in the second 300. Data from the crashed run was discarded; the battles that crashed had been forfeited and would have biased the data towards survivable games.
- `--student` with a relative path failed because the collector changes into the Foul Play directory; the path is now resolved first.
- One run stalled after about 300 battles with no error output (cause not identified). The collector now has a watchdog: a battle with no messages in either direction for 120 s is forfeited, marked as failed (outcome 0) and the last messages are printed; server `|error|` replies are printed (first 20). The watchdog was tested in the sandbox with injected stalls; it has not yet fired on the author's machine (`stalled=0`).

---

### E14: encoder v4 (set-pool beliefs) and PPO that warm-starts from the distilled network (2026-10-13) [sandbox-tested, not yet run on the author's machine]
Built for hypothesis H7 (opponent memory + anchored RL), motivated by the E13 finding that the switch target is the weakest part of the imitation.
- `pokebot/beliefs.py` (new): from Showdown's `data/random-battles/gen9/sets.json` (509 species; 1-3 roles each, with a move pool and Tera types), keep the roles of the opponent's active species compatible with its revealed moves (and ability, if known), and compute for each of OUR six team slots (aligned with switch actions 0-5): worst-case and expected strongest hit from the opponent's possible moves, the expected hit relative to that mon's current HP, the strongest revealed hit, a faster-than-opponent flag and a "worst case could KO now" flag; plus the number of consistent roles, the number of revealed moves, and a Tera-type belief over the 19 types. 58 numbers; encoder **v4** = v3 + these (1025 inputs). Modelling assumptions (roles equally likely; unrevealed pool moves present independently with probability free_slots / pool_size) are an approximation of Showdown's real set generator, not a copy of it. The file path comes from `POKEBOT_SETS`, `data/gen9_random_sets.json`, or a `pokemon-showdown` folder next to the project; if none exists the encoder raises an error (silent zeros would hide a missing file, as happened with the Tera type in E12).
- Prefix compatibility: v2 = first 811 values of v3 = first 811 of v4; `tools/build_dataset.py --encoder v4` rebuilds any recorded dataset from its raw messages, and `pretrain_fp.py` / `train_ppo.py` warm-start from a shorter checkpoint with zero-padded new inputs.
- `train_ppo.py`: `--obs-dim` (811 / 967 / 1025; default = the resumed checkpoint's), warm start from a checkpoint with fewer inputs, KL anchor to a checkpoint that may use an older encoder (it sees only the first columns). Smoke test (128-step rollouts, v2 student, v4 observations, KL coef 0.2): the KL to the anchor started at 0.002 after the first update, i.e. the warm start behaves like the old network.
- `analyze_fp.py` now also prints the switch-target accuracy when both Foul Play and our network switch, next to the random-choice level.
- Checks in the sandbox: replay of 10 battles reproduced masks 290/290 with encoder v4, and the first 967 values equal the stored v3 observation; belief features are non-zero in 39.5% of entries and the Tera belief sums to 1.
- **Signal test (small, sandbox; 120 battles of Foul Play at 60 ms vs the heuristic bot; 552 decisions in which Foul Play switched with at least 2 legal switch targets, 119 battles; 5-fold cross-validation by battle).** A conditional-logit model that scores each of our slots from a few features predicts Foul Play's switch target with accuracy 0.392 (NLL 1.321) using only v2 slot features (opponent STAB threat, our best damage from the bench mon) and 0.430 (NLL 1.287) after adding the 6 belief features per slot; random choice among legal targets is 0.263. A difference of about 4 points on 552 decisions is under two standard errors (about 2 points each), so this is a hint, not a result. Caveats: the sandbox Foul Play used its own stand-in set data, one opponent type, 60 ms search, and the sandbox games are not the author's data.
- Not tested: whether the MLP trained on rebuilt v4 data picks switch targets better (the real test, uses the author's 2,400 battles), and whether anchored PPO from it beats the clone.

---

### E15: v4 (belief features) distillation on the author's 2,400 battles [user, 2026-10-13]
Same five datasets as E13, rebuilt with encoder v4 from the raw messages; `pretrain_fp.py --init checkpoints_v3/fp_best.pt` (new input columns zero-padded), same seed and validation battles (5,352 validation decisions with more than one legal action).
- Training: epoch 0 (before training) val policy_ce 1.872, agreement with the search argmax 0.399; epochs 1-5 val policy_ce 1.877, 1.882, 1.886, 1.901, 1.909 (rising every epoch) while the training loss fell from 1.806 to 1.699; early stop after epoch 5. The best validation loss is therefore at the first epoch, so `fp_best.pt` is essentially the initialisation.
- Analysis (v3 -> v4): overall agreement 0.399 -> 0.394; switch 0.191 -> 0.192; move 0.482 -> 0.474; Tera move 0.000 -> 0.024 (4 of 170); **switch target when both switch: 34.7% (random among legal switches 27.7%, n = 689) -> 34.6% (28.1%, n = 694)**.
- Play (`evaluate.py`, n = 200): v4 0.98 / 0.87 / 0.47 vs random / max-power / heuristic; v3 was 0.98 / 0.81 / 0.45. The differences are inside the +-0.05 to +-0.07 noise at this n and, given that the checkpoint is essentially the v3 initialisation, are not attributable to the features.
- **Reading:** the E14 sandbox hint (switch-target accuracy 0.39 -> 0.43 with beliefs in a small linear model) did not replicate in the MLP on the full data: the MLP did not learn to use the 58 new inputs, and validation loss gets worse with every training epoch. With the training loss falling and the validation loss rising from the first epoch, the network memorises 54k decisions instead of generalising. Possible reasons, not separated: label noise (the search policy is spread in 38% of decisions), too little data for 1,025 inputs and 512 hidden units, features that do not help, or a metric (agreement with a sampled search choice) that has a low ceiling. Untested: stronger regularisation (dropout, higher weight decay, smaller network) or more data with v4; the E14 linear model on the full data.
- Consequence: four distillation runs (E10 11k decisions v2; E11 130k v2; E13 54k v3 + DAgger; E15 54k v4) all give 0.41-0.49 against the heuristic bot with n = 200, i.e. no measurable gain over the behaviour-cloned heuristic. The distillation line is closed for now.

---

### E16: anchored PPO from the distilled v4 network, two KL coefficients (author's machine, 2026-10-13) [user, interim at ~300k steps]
Setup: `train_ppo.py --opponent mixed (20% random / 40% maxpower / 40% heuristic) --resume checkpoints_v4/fp_best.pt --lr 1e-4`, observation v4 (1025 inputs), anchor = the starting checkpoint; run A `--kl-coef 0.2`, run B `--kl-coef 0.05`; about 116-117 steps per second (1M steps is about 2.4 h on the laptop; an earlier estimate of "about a day" was wrong).
- Training logs (sampled actions, last 100 battles per opponent, so about +-0.09 and not comparable to greedy evaluation): at ~178k steps A: max-power 0.59, heuristic 0.25, KL 0.083; B: 0.72, 0.31, KL 0.275. At ~340k steps A: 0.55, 0.25, KL 0.091; B: 0.64, 0.33, KL 0.255. No upward trend against the heuristic bot in either run.
- Greedy evaluation (`evaluate.py`, n = 200, snapshot at 301,056 steps) vs random / max-power / heuristic: **A 0.98 / 0.89 / 0.47; B 0.96 / 0.80 / 0.46.** Starting point (v4 initialisation, E15): 0.98 / 0.87 / 0.47.
- Reading: no improvement over the starting policy in either run. A (strong anchor) stays pinned to the start (KL about 0.09). B (weak anchor) moved further (KL about 0.26); its max-power rate is 7 points lower than the start, about 1.5 standard errors of the difference, so not distinguishable from noise and not evidence of harm either. Same pattern as E7: anchoring keeps the policy, it does not improve it.
- Status: interim. The runs were stopped here by the pre-agreed rule (both at 0.45-0.50 vs the heuristic at 300k steps); they were not run to 1M steps, so a late gain is not excluded.

---

### E17: Foul Play strength vs search budget, against the heuristic bot (author's machine, 2026-10-13) [user]
`tools/collect_foulplay.py --opponent heuristic --search-time-ms {10, 30, 300}`, 200 battles each, three terminals running at the same time (so CPU contention may have lowered the effective search at 300 ms); the 100 ms point is the 500-battle run from E9/E10. The author reported the three result lines without labels; I assigned them to 10 / 30 / 300 ms by their wall-clock (282 s, 446 s, 3241 s for 200 battles), which matches the budgets.

| search time per decision | Foul Play wins | win rate (95% Wilson CI) | wall-clock for the battles |
|---|---|---|---|
| 10 ms | 185 / 200 | 0.925 (0.880-0.954) | 282 s (200 battles) |
| 30 ms | 187 / 200 | 0.935 (0.892-0.962) | 446 s |
| 100 ms | 473 / 500 | 0.946 (0.923-0.963) | 2,877 s (500 battles) |
| 300 ms | 183 / 200 | 0.915 (0.868-0.946) | 3,241 s |

`unmapped` 8, 1 and 5 of 4,722 / 4,352 / 4,621 decisions; `fp_errors` 0 and `stalled` 0 in all three.
- All four intervals overlap: against the heuristic bot Foul Play is already at 0.92-0.95 with 10 ms per decision, and more search shows no measurable gain (300 ms is nominally the lowest). The heuristic bot cannot separate search budgets; the curve is saturated at this opponent.
- For comparison, our network (E15/E16, greedy, n = 200) wins about 0.45-0.47 against the same bot, and against Foul Play at default settings the heuristic bot and the BC clone win 0.06-0.07 (E8). The gap between our network and a search bot that costs about 10 ms per decision is therefore large.
- Consequence for the plan: speed is not a reason to prefer the network at laptop budgets; any benefit of a learned component has to show up as strength against opponents that are not saturated by 10 ms of search, for which the heuristic bot is not a useful probe. Not measured: Foul Play at 1000 ms or more, Foul Play against Foul Play at different budgets, and Foul Play against our network.

---

## 5. Candidate findings (with evidence strength)

| # | Claim | Evidence | Strength |
|---|---|---|---|
| F1 | BC from a scripted teacher with damage-aware features reaches teacher parity in ~1 h (10 min collection + 10 epochs), while 1M steps of PPO from scratch does not (heuristic 0.45 vs 0.19) | E3 vs E5, n=100 each | Moderate: confounds (features, step counts); one seed |
| F2 | PPO fine-tuning from a cloned policy degrades it; a KL anchor prevents the degradation | E6 vs E7, n=200 | Moderate: single seed, single hyperparameter setting |
| F3 | PPO fine-tuning (even anchored) does not improve on the clone within ~1.1M steps in this setup (null result) | E7, n=200 | Moderate: negative results depend on hyperparameters (lr, KL coef, reward) |
| F4 | Damage-aware features help vs max-power | E2 vs E3 | Weak: confounded (H5) |

| F5 | Against Foul Play (search), the heuristic teacher and the BC clone of it are both at about 0.06-0.07 win rate; cloning does not move the ceiling | E8, n=100 each, Foul Play at default search settings | Moderate: single Foul Play configuration; CIs overlap heavily |

| F6 | Distilling Foul Play's search policy into the 811/967-input MLP (up to 54k decisions, Tera features, 600 DAgger battles) leaves heuristic-bot win rate at about 0.45; validation loss overfits in a few epochs; the largest agreement gap is the switch target (about 0.35 correct when a switch is predicted) | E10, E11, E13; evaluate n=200 per run | Moderate for "no gain at this scale and with these features"; single seed; DAgger used a v2 student and beta 0.5 only. Extended by E15: v4 belief features also gave no gain (switch target 34.6% vs 34.7%) |

Not yet evidence for anything: ladder strength, strength vs Metamon, any claim about search or distillation improving our agent.

## 6. Plan and hypotheses (not yet tested)

Decision pending: **format**. Gen 9 random battles (current pipeline; Foul Play supported, no pretrained RL agent) or Gen 9 OU (Metamon's pretrained models and team sets usable; Foul Play also supported; needs team handling and new features such as items/abilities).

- **H7 (chosen 2026-10-13) Opponent memory + anchored RL.** Add features that summarise what has been revealed about the opponent (moves, items, abilities seen; speed bounds; history of the last turns) and, for the opponent's active and bench, set-pool beliefs (H3), then fine-tune with PPO against the heuristic and max-power bots with a KL anchor towards the Foul Play-distilled policy (E7 setup, but starting from the distilled network). Success: heuristic win rate above 0.55 with n >= 500, or a clear gain over the BC clone in switch-target agreement.
- **H1 DAgger with a search teacher.** (E11-E13: plain distillation did not help; DAgger tried once with a v2 student.) Our small net plays; a strong teacher (Foul Play search, or Metamon policy) relabels the states our net visits; retrain; repeat. Success: matches the teacher's win rate against our bot pool at network speed.
- **H2 Reward misalignment.** Test pure win/loss reward vs the shaped reward in the anchored-PPO setting, and/or replace shaping with a learned potential-based shaping from a value model trained on outcomes.
- **H3 Set-pool beliefs.** Random-battle sets come from fixed pools (Showdown `data/random-battles/gen9/sets.json`, 509 species). Feeding the probability that the opponent has each move/item/ability (given what was revealed) should beat teachers that ignore it. Success: BC agent with these features beats 0.45 vs heuristic by more than noise (n >= 1000).
- **H4 Adaptive compute.** Run expensive search only on turns where the net is uncertain or disagrees with a second model.
- **H5 Clean ablation of features.** v0 vs v2 under identical steps, opponent mix and seeds.
- **H6 Evaluation power.** Use >= 1000 games per cell (parallel across cores) and a round-robin with fitted ratings, so differences of ~3 points are detectable.

## 7. Corrections and caveats (keep honest)

- Early on I (Claude) said a competitive agent "typically needs tens of millions" of steps and suggested scaling PPO. E7 at ~1.1M steps shows no gain from more steps; the suggestion was unsupported and has been dropped.
- The first comparison table sent to the author labelled the v0 result "500k steps" for maxpower ~0.5; precisely, that figure is a training-log win rate (sampled actions), not a greedy eval. This log supersedes it (E1/E2).
- Single seed everywhere. No variance across training runs is known.
- Opponents are scripted bots, not humans; none of these numbers say anything about ladder rating.
- Poke-env's `SimpleHeuristicsPlayer` is a fixed baseline; "teacher parity" means parity with it only.
- 2026-10-13: I (Claude) described in E12 the DAgger mode as sandbox-tested with a note that Foul Play's `last_selected_move` effect was unmeasured. On the author's machine it crashed Foul Play in about 44% of DAgger battles (E13). The data from that run was discarded and the mode was fixed. The same note applies to any earlier statement that DAgger data collection "worked".
- 2026-10-13: E14's signal test (0.39 -> 0.43 switch-target accuracy with belief features) was a small sandbox test and I called it a hint; E15 shows it did not carry over to the MLP on the author's data. Treat the E14 number as unreplicated.
- 2026-10-13: I told the author a check (`grep -c FP_BATTLES`) should print "4 or more"; the correct count is 3. No data impact.
- 2026-10-13: agreement numbers across runs use different validation battles and are not strictly comparable (E11 0.447 vs E13 0.399); play-strength numbers use the same evaluation but n = 200 gives about +-0.07.
- Training-time win rates use sampled actions and are not comparable to greedy evaluation numbers.

## 8. Reproduction

```bash
# server
git clone --depth 1 https://github.com/smogon/pokemon-showdown.git && cd pokemon-showdown
npm install && cp config/config-example.js config/config.js
node pokemon-showdown start --no-security          # record: git rev-parse HEAD

# E1-E3 (PPO from scratch)
python train_ppo.py --steps 100000  --opponent random
python train_ppo.py --steps 1000000 --opponent mixed --resume checkpoints/latest.pt

# E4-E5 (teacher data + BC)
python pretrain_bc.py --battles 5000                # -> checkpoints/bc.pt
python evaluate.py --ckpt checkpoints/bc.pt --n 100

# E6 / E7 (PPO from BC)
python train_ppo.py --steps 1000000 --opponent mixed --resume checkpoints/bc.pt --lr 1e-4 --out checkpoints_bc
python train_ppo.py --steps 1000000 --opponent mixed --resume checkpoints/bc.pt --lr 1e-4 \
    --kl-coef 0.2 --ent 0.001 --out checkpoints_kl
python evaluate.py --ckpt checkpoints_kl/step_100352.pt --n 200
```

### Step 0 on the author's machine (Foul Play)
```bash
git clone https://github.com/pmariglia/foul-play && cd foul-play
python -m venv .venv && source .venv/bin/activate     # needs Python 3.11+ and Rust (rustup)
pip install -r requirements.txt                        # builds poke-engine (~1 min)
git apply /path/to/pokebot/tools/foul_play_local_noauth.patch   # required for a local server (see E8)
export FP_LOCAL_NO_AUTH=1

# terminal A: our bot waits (accept mode)
python bench_external.py --opponent FoulPlayBot --bot ckpt --ckpt checkpoints/bc.pt --n 100

# terminal B: Foul Play challenges it (start AFTER terminal A is waiting)
python run.py --websocket-uri ws://localhost:8000/showdown/websocket \
  --ps-username FoulPlayBot --bot-mode challenge_user --user-to-challenge OurBot \
  --pokemon-format gen9randombattle --search-time-ms 200 --search-parallelism 4 \
  --run-count 100 --log-level INFO
# Without the patch + FP_LOCAL_NO_AUTH=1, Foul Play logs "Successfully logged in" but stays a guest
# on a local server and its challenges are ignored (see E8).
```
Record: search time, parallelism, threads, wall-clock per game, and wins/n with the interval printed by the script. Repeat with `--bot heuristic` for a bot-pool reference. (The benchmark runs of 2026-10-07 used no search flags, i.e. the defaults, not the 200 ms / 4 shown above.)

### Recording Foul Play's decisions (E9)
```bash
pip install "poke-env==0.16.1" numpy                 # once, in the Foul Play env (poke2)
node pokemon-showdown start --no-security            # Showdown folder
cd ~/Reinforcement_Learning/pokebot
python tools/collect_foulplay.py --foul-play-dir ./foul-play --battles 20 --out data/fp_test.npz    # sanity check
python tools/collect_foulplay.py --foul-play-dir ./foul-play --battles 500 --out data/fp_500.npz
```
Single terminal, no login patch, nothing else to start. Report `unmapped=` and `fp_errors=` from the progress line.

### v3 data, training and evaluation (E13)
```bash
# poke2 env, Showdown running; four terminals, each its own names and --out (all from the pokebot folder)
python tools/collect_foulplay.py --foul-play-dir ./foul-play --battles 600 --opponent heuristic --fp-name FP1 --opp-name Opp1 --out data/v3_a.npz
python tools/collect_foulplay.py --foul-play-dir ./foul-play --battles 600 --opponent heuristic --fp-name FP2 --opp-name Opp2 --out data/v3_b.npz
python tools/collect_foulplay.py --foul-play-dir ./foul-play --battles 600 --opponent maxpower  --fp-name FP3 --opp-name Opp3 --out data/v3_c.npz
python tools/collect_foulplay.py --foul-play-dir ./foul-play --battles 300 --opponent heuristic --student checkpoints_fp_big/fp_best.pt --beta 0.5 --fp-name FP7  --opp-name Opp7  --out data/v3_d.npz
python tools/collect_foulplay.py --foul-play-dir ./foul-play --battles 300 --opponent heuristic --student checkpoints_fp_big/fp_best.pt --beta 0.5 --fp-name FP10 --opp-name Opp10 --out data/v3_e.npz
# poke env
python pretrain_fp.py --data data/v3_a.npz data/v3_b.npz data/v3_c.npz data/v3_d.npz data/v3_e.npz --init checkpoints_fp_big/fp_best.pt --out checkpoints_v3
python analyze_fp.py  --ckpt checkpoints_v3/fp_best.pt --data data/v3_a.npz data/v3_b.npz data/v3_c.npz data/v3_d.npz data/v3_e.npz
python evaluate.py    --ckpt checkpoints_v3/fp_best.pt --n 200
```
Reuse of a player name after an interrupted run can leave a stale login/battle on the server; use new `--fp-name/--opp-name` after interrupting.

## 9. References

- Metamon: Grigsby et al., "Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers", https://arxiv.org/abs/2504.04395 ; code https://github.com/UT-Austin-RPL/metamon
- PokéAgent Challenge (NeurIPS 2025 competition paper): https://arxiv.org/abs/2603.15563
- PokéChamp: Karten et al., https://arxiv.org/abs/2503.04094
- Foul Play (search bot): https://github.com/pmariglia/foul-play ; engine https://github.com/pmariglia/poke-engine
- poke-env: https://github.com/hsahovic/poke-env
