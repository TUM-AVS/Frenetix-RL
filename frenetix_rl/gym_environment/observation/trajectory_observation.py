__author__ = "Rainer Trauth, Alexander Hobmeier"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Release"

from collections import OrderedDict
from typing import Union, Dict
import numpy as np
from gymnasium.spaces import Box
from cr_scenario_handler.utils.agent_status import AgentStatus
from frenetix_rl.gym_environment.observation import Observation


class TrajectoryObservation(Observation):

    def __init__(self, configs: Dict, configs_name: str = "trajectory_configs"):
        """
        :param configs: dictionary to store all observation configurations
        :param configs_name: key of configs dictionary corresponding to this observation
        """

        trajectory_configs = configs["observation_configs"][configs_name]
        self.observe_feasible_trajectory = trajectory_configs["observe_feasible_percentage"]
        self.observe_is_collision = trajectory_configs["observe_collision"]
        self.observe_not_feasible = trajectory_configs["observe_not_feasible"]
        self.observe_ego_risk = trajectory_configs["observe_ego_risk"]
        self.observe_obstacle_risk = trajectory_configs["observe_obstacle_risk"]

        self.observation_dict = OrderedDict()

    def build_observation_space(self) -> OrderedDict:
        observation_space_dict = OrderedDict()

        if self.observe_feasible_trajectory:
            observation_space_dict["observe_feasible_percentage"] = Box(0, 1, (1,), dtype=np.float32)

        if self.observe_is_collision:
            observation_space_dict["is_collision"] = Box(0, 1, (1,), dtype=np.int8)
        if self.observe_not_feasible:
            observation_space_dict["not_feasible"] = Box(0, 1, (1,), dtype=np.int8)

        if self.observe_ego_risk:
            observation_space_dict["ego_risk"] = Box(0, 1, (1,), dtype=np.float32)
        if self.observe_obstacle_risk:
            observation_space_dict["obstacle_risk"] = Box(0, 1, (1,), dtype=np.float32)

        return observation_space_dict

    def observe(self, agent, simulation, ego_vehicle, optimal_traj) -> Union[np.array, Dict]:

        if self.observe_feasible_trajectory:
            if not agent.all_trajectories:
                self.observation_dict["observe_feasible_percentage"] = np.array([1.0])
            else:
                feasible = sum(1 for traj in agent.all_trajectories if traj.feasible)
                self.observation_dict["observe_feasible_percentage"] = np.array([np.round(feasible/len(agent.all_trajectories), 3)])

        if self.observe_not_feasible:
            if agent.status == AgentStatus.ERROR:
                self.observation_dict["not_feasible"] = np.array([1])
            else:
                self.observation_dict["not_feasible"] = np.array([0])

        if self.observe_is_collision:
            if agent.status == AgentStatus.COLLISION:
                self.observation_dict["is_collision"] = np.array([1])
            else:
                self.observation_dict["is_collision"] = np.array([0])

        if self.observe_ego_risk:
            if optimal_traj is None:
                self.observation_dict["ego_risk"] = np.array([0])
            else:
                self.observation_dict["ego_risk"] = np.array([optimal_traj._ego_risk])

        if self.observe_obstacle_risk:
            if optimal_traj is None:
                self.observation_dict["obstacle_risk"] = np.array([0])
            else:
                self.observation_dict["obstacle_risk"] = np.array([optimal_traj._obst_risk])

        return self.observation_dict



