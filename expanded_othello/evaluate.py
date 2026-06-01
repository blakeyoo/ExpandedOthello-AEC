import numpy as np
from expanded_othello.presets import load_preset as _load_raw
from expanded_othello.agents.random import RandomAgent
from expanded_othello.agents.mcts import MCTSAgent

_OPPONENTS = {"mcts": MCTSAgent, "random": RandomAgent}


def evaluate(agent, opponent="mcts", preset_ids=range(56), n_games=10):
    """Evaluate an agent against a built-in opponent across preset environments.

    Parameters
    ----------
    agent :
        Object with ``act(obs, action_mask) -> int``.
        obs is np.float32 of shape (3, H, W) — channel 0: my discs,
        channel 1: opponent discs, channel 2: obstacles.
        action_mask is np.int8 of length H*W (1 = legal move).
    opponent : {"mcts", "random"}
        Built-in opponent to play against.
    preset_ids : range or list of int
        Preset indices to evaluate on (0–55).
    n_games : int
        Number of games per environment. Colors alternate each game
        so results are balanced across first-move advantage.

    Returns
    -------
    dict
        {preset_id: {"win_rate": float, "draw_rate": float, "loss_rate": float}}
    """
    if opponent not in _OPPONENTS:
        raise ValueError(f"Unknown opponent '{opponent}'. Choose from {list(_OPPONENTS)}.")

    results = {}
    for pid in preset_ids:
        wins = draws = losses = 0
        for i in range(n_games):
            user_color = 1 if i % 2 == 0 else -1
            env = _load_raw(pid)
            opp = _OPPONENTS[opponent](player=-user_color)
            winner = _play(agent, opp, env, user_color)
            if winner == user_color:
                wins += 1
            elif winner == 0:
                draws += 1
            else:
                losses += 1
        results[pid] = {
            "win_rate":  wins   / n_games,
            "draw_rate": draws  / n_games,
            "loss_rate": losses / n_games,
        }
    return results


def _play(user_agent, builtin_agent, env, user_color):
    w = env.board_width
    while not env.is_done():
        if env.current_player == user_color:
            moves = env.valid_moves(user_color)
            if not moves:
                env.current_player *= -1
                continue
            obs  = _obs(env, user_color)
            mask = _mask(env, user_color)
            action = user_agent.act(obs, mask)
            x, y = divmod(int(action), w)
            if env.is_valid_move(x, y, user_color):
                env.step(x, y)
            else:
                env.current_player *= -1
        else:
            move = builtin_agent.select_action(env)
            if move is not None:
                env.step(*move)
            else:
                env.current_player *= -1
    return env.get_winner()


def _obs(env, player):
    my, opp = (env.board[0], env.board[1]) if player == 1 else (env.board[1], env.board[0])
    return np.stack([my, opp, env.board[2]], axis=0)


def _mask(env, player):
    h, w = env.board_height, env.board_width
    mask = np.zeros(h * w, dtype=np.int8)
    for r, c in env.valid_moves(player):
        mask[r * w + c] = 1
    return mask
