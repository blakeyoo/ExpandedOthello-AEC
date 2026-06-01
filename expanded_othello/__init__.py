"""Expanded Othello AI Arena — parametric benchmark with variable board geometry and win conditions.

Quick start
-----------
from expanded_othello import make_env, load_preset
from expanded_othello.agents import RandomAgent, MCTSAgent

# PettingZoo AEC interface (primary)
env = make_env(board_size=8, win_cond=0.6)
env.reset()
for agent in env.agent_iter():
    obs, reward, terminated, truncated, info = env.last()
    action = None if (terminated or truncated) else my_agent.act(obs, info["action_mask"])
    env.step(action)

# Load one of the 56 official test environments (index 0–55)
env = load_preset(3)

# Raw OthelloEnv (for custom agent loops)
from expanded_othello.env import OthelloEnv
raw = OthelloEnv(board_size=8, win_cond=0.6)
winner = raw.play_game(RandomAgent(1), MCTSAgent(-1))
"""

from expanded_othello.env import OthelloEnv
from expanded_othello.wrappers.aec import OthelloAECEnv
from expanded_othello.presets import PRESETS, load_preset as _load_preset_raw
from expanded_othello.evaluate import evaluate


def make_env(
    board_size=8,
    obstacles=None,
    win_cond: float = 1.01,
    n_turns: int = -1,
    turn_rule: str = "default",
) -> OthelloAECEnv:
    """Create a fresh OthelloAECEnv (PettingZoo AEC) with the given parameters."""
    raw = OthelloEnv(
        board_size=board_size,
        obstacles=obstacles,
        win_cond=win_cond,
        n_turns=n_turns,
        turn_rule=turn_rule,
    )
    return OthelloAECEnv(raw)


def load_preset(index: int) -> OthelloAECEnv:
    """Return a fresh OthelloAECEnv for the given preset index (0–55)."""
    return OthelloAECEnv(_load_preset_raw(index))


__all__ = ["OthelloAECEnv", "OthelloEnv", "make_env", "load_preset", "PRESETS", "evaluate"]
