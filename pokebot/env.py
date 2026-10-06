"""Gymnasium env: poke-env SinglesEnv + our observation / reward.

poke-env (0.16) returns {"observation": embed_battle(...), "action_mask": ...}
from reset()/step(), so embed_battle only has to return the feature vector.
"""
from __future__ import annotations

import numpy as np
from gymnasium.spaces import Box, Dict
from poke_env.environment import SinglesEnv, SingleAgentWrapper
from poke_env.player import Player, RandomPlayer

from .encoder import OBS_DIM, encode_battle


class PokemonEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        n_actions = self.action_spaces[self.possible_agents[0]].n
        space = Dict(
            {
                "observation": Box(-1.0, 4.0, shape=(OBS_DIM,), dtype=np.float32),
                "action_mask": Box(0, 1, shape=(n_actions,), dtype=np.int8),
            }
        )
        self.observation_spaces = {a: space for a in self.possible_agents}

    def embed_battle(self, battle):
        return encode_battle(battle)

    def calc_reward(self, battle) -> float:
        # Light shaping on top of the +/-1 win signal. Anneal these toward 0
        # later so the final policy optimises winning, not HP.
        return self.reward_computing_helper(
            battle, fainted_value=0.15, hp_value=0.05, victory_value=1.0
        )


def make_env(
    opponent: Player | None = None,
    battle_format: str = "gen9randombattle",
    **kwargs,
) -> SingleAgentWrapper:
    """Single-agent view: the other side is driven by `opponent` (a poke-env Player)."""
    # strict=False: if poke-env can't map an order to an action (rare edge cases such as a
    # locked move like Outrage on a mon with more than 4 known moves), fall back to a random
    # legal move instead of killing a multi-hour training run.
    kwargs.setdefault("strict", False)
    env = PokemonEnv(battle_format=battle_format, **kwargs)
    return SingleAgentWrapper(env, opponent or RandomPlayer(start_listening=False))
