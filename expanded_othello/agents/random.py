import random as _random


class RandomAgent:
    def __init__(self, player: int = 1):
        self.player = player

    def select_action(self, env):
        moves = env.valid_moves(self.player)
        return _random.choice(moves) if moves else None
