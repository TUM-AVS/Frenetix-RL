"""Abstract class for rewards"""

from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np


class Reward(ABC):
    """Abstract class for rewards"""

    def reset(self, test_env, cost_weights):
        pass

    @abstractmethod
    def calc_reward(self, observation_dict: OrderedDict, ego_action: np.ndarray, agent, config_sim=None,
                    simulation=None, termination_info=None) -> list:
        """
        Calculate the reward according to the observations

        :param observation_dict: current observations
        :param ego_action: Current ego_action of the environment
        :param agent: agent information class
        :param config_sim: config information (optional)
        :param simulation: simulation information (optional)
        :param termination_info: Info messages on termination
        :return: list of all rewards
        """

