import numpy as np
import copy
from numba import njit


# ---------------------------------------------------------------------------
# JIT-compiled core functions (module-level, no Python objects)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _is_valid_move(board, h, w, x, y, player):
    if x < 0 or x >= h or y < 0 or y >= w:
        return False
    if board[0, x, y] + board[1, x, y] + board[2, x, y] > 0:
        return False

    my_ch  = 0 if player == 1 else 1
    opp_ch = 1 - my_ch

    dirs = ((-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1))
    for i in range(8):
        dx = dirs[i][0]; dy = dirs[i][1]
        nx = x + dx;     ny = y + dy
        found = False
        while 0 <= nx < h and 0 <= ny < w and board[opp_ch, nx, ny] == 1.0:
            nx += dx; ny += dy
            found = True
        if found and 0 <= nx < h and 0 <= ny < w and board[my_ch, nx, ny] == 1.0:
            return True
    return False


@njit(cache=True)
def _valid_moves(board, h, w, player):
    """Returns (N, 2) int64 array of valid (row, col) moves."""
    buf = np.empty((h * w, 2), dtype=np.int64)
    count = 0
    for x in range(h):
        for y in range(w):
            if _is_valid_move(board, h, w, x, y, player):
                buf[count, 0] = x
                buf[count, 1] = y
                count += 1
    return buf[:count]


@njit(cache=True)
def _step(board, h, w, x, y, player):
    """Apply move (x, y) for player on board in-place. Assumes move is valid."""
    my_ch  = 0 if player == 1 else 1
    opp_ch = 1 - my_ch

    board[my_ch, x, y] = 1.0

    dirs  = ((-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1))
    flips = np.empty((h + w, 2), dtype=np.int64)

    for i in range(8):
        dx = dirs[i][0]; dy = dirs[i][1]
        nx = x + dx;     ny = y + dy
        flip_count = 0
        while 0 <= nx < h and 0 <= ny < w and board[opp_ch, nx, ny] == 1.0:
            flips[flip_count, 0] = nx
            flips[flip_count, 1] = ny
            flip_count += 1
            nx += dx; ny += dy
        if flip_count > 0 and 0 <= nx < h and 0 <= ny < w and board[my_ch, nx, ny] == 1.0:
            for j in range(flip_count):
                fx = flips[j, 0]; fy = flips[j, 1]
                board[opp_ch, fx, fy] = 0.0
                board[my_ch,  fx, fy] = 1.0


@njit(cache=True)
def _get_winner(board, win_cond):
    black = np.sum(board[0])
    white = np.sum(board[1])
    total = black + white
    k     = win_cond

    if total == 0.0 or k == 0.5:
        return 0

    if k > 0.5:
        winner = 1 if black > white else (-1 if white > black else 0)
    else:
        winner = 1 if black < white else (-1 if white < black else 0)

    if winner == 0:
        return 0

    win_count = black if winner == 1 else white
    ratio = win_count / total

    if ratio == 0.5 or (k > 0.5 and ratio >= k) or (k < 0.5 and ratio <= k):
        return 0

    return winner


# ---------------------------------------------------------------------------
# OthelloEnv — thin Python wrapper around the JIT functions
# ---------------------------------------------------------------------------

class OthelloEnv:
    """Parametric Othello environment. E = (L, C) where L is board geometry and C is win condition."""

    def __init__(self, board_size=8, obstacles=None, turn_rule="default", win_cond=1.01, n_turns=-1):
        if isinstance(board_size, int):
            self.board_height = self.board_width = board_size
        elif isinstance(board_size, (tuple, list)) and len(board_size) == 2:
            self.board_height, self.board_width = board_size
        else:
            raise ValueError("board_size must be int or (rows, cols)")

        if turn_rule not in {"default", "fewer_consecutive"}:
            raise ValueError(f"Unknown turn_rule '{turn_rule}'. Use 'default' or 'fewer_consecutive'.")

        self.win_cond          = float(win_cond)
        self.turn_rule         = turn_rule
        self.n_turns           = n_turns
        self.turn              = 0
        self.current_player    = 1
        self.consecutive_count = 0

        self.board = np.zeros((3, self.board_height, self.board_width), dtype=np.float32)

        if obstacles:
            for x, y in obstacles:
                self.board[2, x, y] = 1.0

        cx, cy = self.board_height // 2 - 1, self.board_width // 2 - 1
        for (dx, dy), color in [((0,0), 1), ((1,1), 1), ((0,1), -1), ((1,0), -1)]:
            x, y = cx + dx, cy + dy
            if 0 <= x < self.board_height and 0 <= y < self.board_width and self.board[2, x, y] == 0:
                self.board[0 if color == 1 else 1, x, y] = 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_valid_move(self, x, y, player=None) -> bool:
        p = player if player is not None else self.current_player
        return bool(_is_valid_move(self.board, self.board_height, self.board_width, x, y, p))

    def valid_moves(self, player=None) -> list:
        p   = player if player is not None else self.current_player
        arr = _valid_moves(self.board, self.board_height, self.board_width, p)
        return [(int(arr[i, 0]), int(arr[i, 1])) for i in range(len(arr))]

    def step(self, x, y) -> bool:
        if not self.is_valid_move(x, y):
            return False

        _step(self.board, self.board_height, self.board_width, x, y, self.current_player)
        self.turn += 1

        if self.turn_rule == "default":
            self.current_player *= -1
        elif self.turn_rule == "fewer_consecutive":
            black     = float(np.sum(self.board[0]))
            white     = float(np.sum(self.board[1]))
            my_count  = black if self.current_player == 1 else white
            opp_count = white if self.current_player == 1 else black
            if my_count < opp_count:
                self.consecutive_count += 1
                if self.consecutive_count >= 2:
                    self.consecutive_count = 0
                    self.current_player *= -1
            else:
                self.consecutive_count = 0
                self.current_player *= -1

        return True

    def is_done(self) -> bool:
        if self.n_turns > 0 and self.turn >= self.n_turns:
            return True
        h, w = self.board_height, self.board_width
        return (len(_valid_moves(self.board, h, w,  1)) == 0 and
                len(_valid_moves(self.board, h, w, -1)) == 0)

    def get_winner(self) -> int:
        return int(_get_winner(self.board, self.win_cond))

    def get_board(self) -> np.ndarray:
        """Returns (3, H, W) float32 array: [my_discs, opp_discs, obstacles]."""
        if self.current_player == 1:
            my_plane, opp_plane = self.board[0], self.board[1]
        else:
            my_plane, opp_plane = self.board[1], self.board[0]
        return np.stack([my_plane, opp_plane, self.board[2]], axis=0)

    def clone(self) -> "OthelloEnv":
        return copy.deepcopy(self)

    def get_next_state(self, x, y) -> "OthelloEnv":
        c = self.clone()
        c.step(x, y)
        return c

    def play_game(self, agent_black, agent_white) -> int:
        """Run a full game. Returns winner: 1 (black), -1 (white), 0 (draw)."""
        while not self.is_done():
            agent = agent_black if self.current_player == 1 else agent_white
            move  = agent.select_action(self)
            if move is not None:
                self.step(*move)
            else:
                self.current_player *= -1
        return self.get_winner()
