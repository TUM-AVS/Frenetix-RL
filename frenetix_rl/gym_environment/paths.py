"""
Module for CommonRoad Gym environment related constants
"""
from pathlib import Path
# Path
ROOT_STR = str(Path(__file__).parent.parent.parent)


PATH_PARAMS = {
    "root": ROOT_STR,
    "hyperparams": ROOT_STR + "/frenetix_rl/hyperparams/ppo2.yml",
    "logs": ROOT_STR + "/logs",
    "logs_tensorboard": ROOT_STR + "/logs_tensorboard",
    "configs": ROOT_STR + "/frenetix_rl/gym_environment/configs.yaml",
    "planner_configs": ROOT_STR + "/configurations"
}
