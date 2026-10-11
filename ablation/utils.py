import json
import os
from typing import Any, Dict

from constant import (
    DEFAULT_SEED, SIM_TIME, NUM_WORKSTATIONS, LAMBDA_GRID, NUM_ROBOTS,
    ROBOT_CAPACITY, PARTIAL_K, TOTAL_EPISODES, N_EVAL,
)

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed":              DEFAULT_SEED,
    "sim_time":          SIM_TIME,
    "num_workstations":  NUM_WORKSTATIONS,
    "lambda_default":    800 / 3600,
    "lambda_grid":       list(LAMBDA_GRID),
    "robot_counts":      [NUM_ROBOTS],
    "robot_capacity":    ROBOT_CAPACITY,
    "partial_k":         PARTIAL_K,
    "total_episodes":    TOTAL_EPISODES,
    "n_eval":            N_EVAL,
}


def load_config(path: str = "config.json") -> Dict[str, Any]:
    """JSON 설정 파일을 읽어 DEFAULT_CONFIG 위에 덮어씀

    입력:
        path: 설정 파일 경로
    출력:
        설정 딕셔너리 (파일이 없으면 DEFAULT_CONFIG 복사본)
    """
    if os.path.exists(path):
        with open(path) as f:
            return {**DEFAULT_CONFIG, **json.load(f)}
    return DEFAULT_CONFIG.copy()
