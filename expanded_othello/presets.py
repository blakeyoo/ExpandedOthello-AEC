"""56 official test environments from the Expanded Othello paper."""
from expanded_othello.env import OthelloEnv

_BOARD_CONFIGS = [
    {"board_size": 8,        "obstacles": [],                                                                                                                                                                                                                                                                                      "name": "Standard 8×8"},
    {"board_size": 8,        "obstacles": [(0,0),(0,7),(7,0),(7,7)],                                                                                                                                                                                                                                                               "name": "No Corners"},
    {"board_size": 8,        "obstacles": [(0,1),(1,7),(6,0),(7,6)],                                                                                                                                                                                                                                                               "name": "Partial C-Squares"},
    {"board_size": 8,        "obstacles": [(1,1),(1,6),(6,1),(6,6)],                                                                                                                                                                                                                                                               "name": "X-Squares"},
    {"board_size": (12, 10), "obstacles": [(0,1),(11,1),(2,4),(1,2),(2,7),(9,2),(9,5),(2,2),(0,8),(2,5),(11,8),(9,7),(9,3),(9,6),(10,7),(2,3),(1,7),(5,0),(2,6),(6,0),(5,9),(6,9),(10,2),(9,4)],                                                                                                                                    "name": "Random Board 1 (12×10)"},
    {"board_size": (6, 8),   "obstacles": [(1,2),(2,1),(1,5),(3,1),(4,2),(4,5),(2,6),(3,6)],                                                                                                                                                                                                                                       "name": "Random Board 2 (6×8)"},
    {"board_size": (10, 10), "obstacles": [(0,7),(4,0),(2,1),(4,9),(9,2),(7,3),(0,2),(7,6),(2,8),(9,0),(7,1),(9,9),(0,0),(0,9),(2,3),(5,0),(2,6),(5,9),(9,7),(7,8)],                                                                                                                                                               "name": "Random Board 3 (10×10)"},
]

_WIN_CONDITIONS = [
    {"win_cond": 1.01,  "n_turns": -1, "label": "Majority"},
    {"win_cond": -0.01, "n_turns": -1, "label": "Minority"},
    {"win_cond": 0.8,   "n_turns": -1, "label": "Majority<80%"},
    {"win_cond": 0.6,   "n_turns": -1, "label": "Majority<60%"},
    {"win_cond": 0.4,   "n_turns": -1, "label": "Minority>40%"},
    {"win_cond": 0.2,   "n_turns": -1, "label": "Minority>20%"},
    {"win_cond": 1.01,  "n_turns": 10, "label": "Majority/Blitz"},
    {"win_cond": -0.01, "n_turns": 10, "label": "Minority/Blitz"},
]

PRESETS: list[dict] = []
_preset_id = 0
for wc in _WIN_CONDITIONS:
    for board in _BOARD_CONFIGS:
        PRESETS.append({
            "id":        _preset_id,
            "name":      f"{board['name']} | {wc['label']}",
            "board_size": board["board_size"],
            "obstacles": board["obstacles"],
            "win_cond":  wc["win_cond"],
            "n_turns":   wc["n_turns"],
            "turn_rule": "default",
        })
        _preset_id += 1


def load_preset(index: int, turn_rule: str = "default") -> OthelloEnv:
    """Return a fresh OthelloEnv for the given preset index (0–55)."""
    if not 0 <= index < len(PRESETS):
        raise IndexError(f"Preset index must be 0–{len(PRESETS) - 1}, got {index}.")
    p = PRESETS[index]
    return OthelloEnv(
        board_size=p["board_size"],
        obstacles=p["obstacles"],
        win_cond=p["win_cond"],
        n_turns=p["n_turns"],
        turn_rule=turn_rule,
    )
