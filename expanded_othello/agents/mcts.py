import math
from collections import defaultdict
import numpy as np
from numba import njit

from expanded_othello.env import _valid_moves, _step, _get_winner


@njit(cache=True)
def _mcts_rollout(board, h, w, win_cond, player, turn, n_turns, rollout_depth):
    """JIT-compiled random rollout. Returns winner (1, -1, or 0)."""
    board = board.copy()
    depth = 0
    while depth < rollout_depth:
        if n_turns > 0 and turn >= n_turns:
            break
        my_moves = _valid_moves(board, h, w, player)
        if len(my_moves) == 0:
            if len(_valid_moves(board, h, w, -player)) == 0:
                break
            player = -player
            continue
        idx = np.random.randint(0, len(my_moves))
        _step(board, h, w, int(my_moves[idx, 0]), int(my_moves[idx, 1]), player)
        turn  += 1
        player = -player
        depth += 1
    return _get_winner(board, win_cond)


class MCTSAgent:
    """MCTS agent. Knows the win condition (oracle by design)."""

    def __init__(self, player: int, num_simulations: int = 100,
                 exploration_weight: float = 1.4, rollout_depth: int = 30):
        self.player = player
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.rollout_depth = rollout_depth

    def select_action(self, env):
        h, w        = env.board_height, env.board_width
        root_board  = env.board.copy()
        root_player = env.current_player
        root_turn   = env.turn
        c           = self.exploration_weight

        visits: dict = defaultdict(int)
        wins:   dict = defaultdict(float)

        def board_key(board, player, move):
            return (hash(board[0].tobytes()), hash(board[1].tobytes()), player, move)

        def ucb(parent_total, w_val, v):
            if v == 0:
                return math.inf
            return w_val / v + c * math.sqrt(math.log(parent_total) / v)

        for _ in range(self.num_simulations):
            board  = root_board.copy()
            player = root_player
            turn   = root_turn
            path   = []

            # Selection + Expansion
            while True:
                moves_arr = _valid_moves(board, h, w, player)
                if len(moves_arr) == 0:
                    if len(_valid_moves(board, h, w, -player)) == 0:
                        break
                    player = -player
                    continue

                moves = [(int(moves_arr[i, 0]), int(moves_arr[i, 1]))
                         for i in range(len(moves_arr))]
                parent_total = 1 + sum(visits[board_key(board, player, m)] for m in moves)

                best_move  = moves[0]
                best_score = -math.inf
                for m in moves:
                    score = ucb(parent_total,
                                wins[board_key(board, player, m)],
                                visits[board_key(board, player, m)])
                    if score > best_score:
                        best_score = score
                        best_move  = m

                key = board_key(board, player, best_move)
                path.append(key)
                _step(board, h, w, best_move[0], best_move[1], player)
                turn  += 1
                player = -player

                if visits[key] == 0:
                    break

            # Rollout (JIT)
            result = _mcts_rollout(board, h, w, env.win_cond, player, turn,
                                   env.n_turns, self.rollout_depth)

            # Backpropagation
            for key in path:
                visits[key] += 1
                if result == self.player:
                    wins[key] += 1
                elif result == 0:
                    wins[key] += 0.5

        root_moves_arr = _valid_moves(root_board, h, w, root_player)
        if len(root_moves_arr) == 0:
            return None
        root_moves = [(int(root_moves_arr[i, 0]), int(root_moves_arr[i, 1]))
                      for i in range(len(root_moves_arr))]
        return max(root_moves, key=lambda m: visits[board_key(root_board, root_player, m)])
