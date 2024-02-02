import logging
from collections import OrderedDict
from typing import Union, Dict

import gymnasium as gym
import numpy as np

from frenetix_rl.gym_environment.observation.ego_observation import EgoObservation
from frenetix_rl.gym_environment.observation.goal_observation import GoalObservation
from frenetix_rl.gym_environment.observation.surrounding_observation import SurroundingObservation
from frenetix_rl.gym_environment.observation.cost_observation import CostObservation
from frenetix_rl.gym_environment.observation.trajectory_observation import TrajectoryObservation

LOGGER = logging.getLogger(__name__)


class ObservationCollector:
    """
    This class is a wrapper for individual observation classes. It serves as an access point for the observations.
    Currently, this class supports the GoalObservation, RPSurroundingObservation, LaneletNetworkObservation,
    EgoObservation, CostObservation and TrajectoryObservation.

    :param configs: dictionary with the parameters of the configuration
        should include an individual dictionary for the individual observation classes
    """

    def __init__(self, configs: Dict):
        self._flatten_observation = configs.get("flatten_observation")

        self._scenario = None
        self._planning_problem = None
        self.observation_dict = None

        self.ego_observation = EgoObservation(configs)
        self.goal_observation = GoalObservation(configs)
        self.surrounding_observation = SurroundingObservation(configs)
        self.cost_observation = CostObservation(configs)
        self.trajectory_observation = TrajectoryObservation(configs)

        self.observation_space = self._build_observation_space()

    def _build_observation_space(self) -> Union[gym.spaces.Box, gym.spaces.Dict]:
        """
        builds the observation space dictionary

        :return: the function returns an OrderedDict with the observation spaces of each observation as an entry
        """

        observation_space_dict = OrderedDict()
        observation_space_dict.update(self.ego_observation.build_observation_space())
        observation_space_dict.update(self.goal_observation.build_observation_space())
        observation_space_dict.update(self.surrounding_observation.build_observation_space())
        observation_space_dict.update(self.cost_observation.build_observation_space())
        observation_space_dict.update(self.trajectory_observation.build_observation_space())

        self.observation_space_dict = observation_space_dict

        if self._flatten_observation:
            lower_bounds, upper_bounds = np.array([]), np.array([])
            for space in observation_space_dict.values():
                lower_bounds = np.concatenate((lower_bounds, space.low))
                upper_bounds = np.concatenate((upper_bounds, space.high))
            self.observation_space_size = lower_bounds.shape[0]
            observation_space = gym.spaces.Box(low=lower_bounds, high=upper_bounds, dtype=np.float64)
            LOGGER.debug(f"Size of flattened observation space: {self.observation_space_size}")
        else:
            observation_space = gym.spaces.Dict(self.observation_space_dict)
            LOGGER.debug(f"Length of dictionary observation space: {len(self.observation_space_dict)}")

        return observation_space

    def observe(self, agent_env) -> Union[np.array, Dict]:
        # initialize observation_dict
        observation_dict = OrderedDict()

        # The reactive planner which is needed is wrapped in the PlannerInterface of the Agent of the AgentEnv
        agent = agent_env.simulation.agents[0]
        planner = agent.planner_interface.planner
        optimal_traj = planner.optimal_trajectory

        # execute the observations
        observation_dict_ego = self.ego_observation.observe(agent.vehicle_history[-1].initial_state, optimal_traj, planner)
        observation_dict_goal = self.goal_observation.observe(planner, agent.vehicle_history[-1].initial_state, agent, agent_env.status)
        observation_dict_surrounding = self.surrounding_observation.observe(self._scenario, agent.vehicle_history[-1].initial_state, optimal_traj)
        observation_dict_cost = self.cost_observation.observe(agent, optimal_traj)
        observation_dict_trajectory = self.trajectory_observation.observe(agent, agent_env.simulation, agent.vehicle_history[-1],
                                                                          optimal_traj)

        # update observation dictionary
        observation_dict.update(observation_dict_ego)
        observation_dict.update(observation_dict_goal)
        observation_dict.update(observation_dict_surrounding)
        observation_dict.update(observation_dict_cost)
        observation_dict.update(observation_dict_trajectory)

        assert len(list(observation_dict.keys())) == len(list(self.observation_space_dict.keys()))
        self.observation_dict = OrderedDict((k, observation_dict[k]) for k in self.observation_space_dict.keys())
        assert list(self.observation_dict.keys()) == list(self.observation_space_dict.keys())

        if self._flatten_observation:
            observation_vector = np.zeros(self.observation_space.shape)
            index = 0
            for k in self.observation_dict.keys():
                size = np.prod(self.observation_dict[k].shape)
                observation_vector[index: index + size] = self.observation_dict[k].flat
                index += size
            return observation_vector
        else:
            return self.observation_dict

    def reset(self, agent):

        self.goal_observation.reset(agent.planning_problem, agent)
        self._scenario = agent.scenario
        self._planning_problem = agent.planning_problem

        self.goal_observation.observation_history_dict = dict()
