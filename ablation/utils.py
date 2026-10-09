import json
import os
from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed":              42,
    "sim_time":          36000,
    "num_workstations":  3,
    "lambda_default":    800 / 3600,
    # λ 그리드: 700 / 800 / 900 / 1000 (orders/hr)
    "lambda_grid":       [700 / 3600, 800 / 3600, 900 / 3600, 1000 / 3600],
    "robot_counts":      [14],
    "robot_capacity":    5,
    "partial_k":         3,
    "total_episodes":    1000,
    "n_eval":            20,
}


def load_config(path: str = "config.json") -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path) as f:
            return {**DEFAULT_CONFIG, **json.load(f)}
    return DEFAULT_CONFIG.copy()