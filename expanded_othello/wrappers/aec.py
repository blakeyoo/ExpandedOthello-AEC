"""PettingZoo AEC wrapper for OthelloEnv.

Install extras:  pip install expanded-othello[pettingzoo]
"""
import numpy as np

try:
    from pettingzoo import AECEnv
    from pettingzoo.utils import AgentSelector
    from gymnasium import spaces
except ImportError as e:
    raise ImportError(
        "PettingZoo wrapper requires pettingzoo and gymnasium. "
        "Install with: pip install pettingzoo gymnasium"
    ) from e

from expanded_othello.env import OthelloEnv


class OthelloAECEnv(AECEnv):
    """PettingZoo AEC environment for Expanded Othello.

    Agents:  "black" (player 1) and "white" (player -1).
    Observation: np.float32 array of shape (3, H, W)
                 channels: [my_discs, opponent_discs, obstacles]
    Action:  int in [0, H*W), row-major flat index.
    Info:    {"action_mask": np.int8 array of length H*W}

    Example
    -------
    from expanded_othello import make_env
    from expanded_othello.wrappers import OthelloAECEnv

    env = OthelloAECEnv(make_env(board_size=8, win_cond=0.6))
    env.reset()
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            action = None
        else:
            action = my_agent.act(obs, info["action_mask"])
        env.step(action)
    """

    metadata = {"render_modes": [], "name": "expanded_othello_v0", "is_parallelizable": False}

    def __init__(self, othello_env: OthelloEnv):
        super().__init__()
        self._proto = othello_env
        self.possible_agents = ["black", "white"]
        self._to_player = {"black": 1, "white": -1}
        self._to_agent = {1: "black", -1: "white"}

        h, w = othello_env.board_height, othello_env.board_width
        obs_space = spaces.Box(0.0, 1.0, shape=(3, h, w), dtype=np.float32)
        act_space = spaces.Discrete(h * w)
        self.observation_spaces = {a: obs_space for a in self.possible_agents}
        self.action_spaces = {a: act_space for a in self.possible_agents}

    # ------------------------------------------------------------------
    # PettingZoo interface
    # ------------------------------------------------------------------

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        obs_indices = np.argwhere(self._proto.board[2] == 1.0)
        obstacles = [(int(x), int(y)) for x, y in obs_indices]
        self._env = OthelloEnv(
            board_size=(self._proto.board_height, self._proto.board_width),
            obstacles=obstacles,
            win_cond=self._proto.win_cond,
            turn_rule=self._proto.turn_rule,
            n_turns=self._proto.n_turns,
        )

        self.agents = self.possible_agents[:]
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {"action_mask": self._action_mask(a)} for a in self.agents}

        self.agent_selection = self._to_agent[self._env.current_player]

    def observe(self, agent):
        player = self._to_player[agent]
        my, opp = (self._env.board[0], self._env.board[1]) if player == 1 \
                  else (self._env.board[1], self._env.board[0])
        return np.stack([my, opp, self._env.board[2]], axis=0)

    def step(self, action):
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._clear_rewards()

        player = self._to_player[agent]
        w = self._env.board_width

        if action is not None and self._env.valid_moves(player):
            x, y = divmod(int(action), w)
            self._env.step(x, y)
        else:
            self._env.current_player *= -1

        # Skip over players with no moves (consecutive pass handling)
        for _ in range(2):
            if self._env.is_done():
                break
            if not self._env.valid_moves(self._env.current_player):
                self._env.current_player *= -1

        if self._env.is_done():
            winner = self._env.get_winner()
            for a in self.agents:
                p = self._to_player[a]
                self.rewards[a] = float(p * winner) if winner != 0 else 0.0
                self.terminations[a] = True
            self.agent_selection = self._agent_selector.next()
        else:
            self.agent_selection = self._to_agent[self._env.current_player]

        self._accumulate_rewards()
        self.infos = {a: {"action_mask": self._action_mask(a)} for a in self.agents}

    def render(self):
        pass

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _action_mask(self, agent):
        player = self._to_player[agent]
        h, w = self._env.board_height, self._env.board_width
        mask = np.zeros(h * w, dtype=np.int8)
        for r, c in self._env.valid_moves(player):
            mask[r * w + c] = 1
        return mask
