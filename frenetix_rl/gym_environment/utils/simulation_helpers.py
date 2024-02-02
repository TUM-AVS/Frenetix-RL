import numpy as np
from gymnasium.spaces import Box

from commonroad.scenario.obstacle import ObstacleType


def select_agents(scenario):
    """ Selects the dynamic obstacles that should be simulated as agents
    according to the multiagent configuration.

    :return: A List of obstacle IDs that can be used as agents
    """

    # Find all dynamic obstacles in the scenario
    allowed_types = [ObstacleType.CAR,
                     ObstacleType.TRUCK,
                     ObstacleType.BUS]
    all_obs_ids = list(filter(lambda id: scenario.obstacle_by_id(id).obstacle_type in allowed_types,
                              [obs.obstacle_id for obs in scenario.obstacles]))
    return all_obs_ids


def construct_action_space(size) -> Box:
    return Box(low=-1, high=1, shape=(size,), dtype=np.float64)
