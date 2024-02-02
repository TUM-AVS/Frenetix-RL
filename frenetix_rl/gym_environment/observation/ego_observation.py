"""
Module EgoObservation
"""
from collections import defaultdict, OrderedDict
from typing import Union, Dict
import gymnasium as gym
import numpy as np
from numpy import ndarray

from frenetix_rl.gym_environment.observation.observation import Observation
from cr_scenario_handler.utils.helper_functions import calc_orientation_of_line


class EgoObservation(Observation):
    """
    Ego-vehicle-related observation class
    """

    def __init__(self, configs: Dict, configs_name: str = "ego_configs"):
        """

        :param configs: dictionary to store all observation configurations
        :param configs_name: key of configs dictionary corresponding to this observation
        """

        # Read config
        ego_configs = configs["observation_configs"][configs_name]
        self.observe_v_ego: bool = ego_configs.get("observe_v_ego")
        self.observe_a_ego: bool = ego_configs.get("observe_a_ego")
        self.observe_jerk_lat_ego: bool = ego_configs.get("observe_jerk_lat_ego")
        self.observe_jerk_long_ego: bool = ego_configs.get("observe_jerk_long_ego")
        self.observe_relative_orientation: bool = ego_configs.get("observe_relative_orientation")
        self.observe_steering_angle: bool = ego_configs.get("observe_steering_angle")
        self.observe_yaw_rate: bool = ego_configs.get("observe_yaw_rate")
        self.observe_lat_diff_ref_path: bool = ego_configs.get("observe_lat_diff_ref_path")

        self.observation_dict = OrderedDict()
        self.observation_history_dict = defaultdict(list)

    def build_observation_space(self) -> OrderedDict:
        observation_space_dict = OrderedDict()

        if self.observe_v_ego:
            observation_space_dict["v_ego"] = gym.spaces.Box(0.0, 50, (1,), dtype=np.float32)
        if self.observe_a_ego:
            observation_space_dict["a_ego"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        if self.observe_jerk_lat_ego:
            observation_space_dict["jerk_lat_ego"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        if self.observe_jerk_long_ego:
            observation_space_dict["jerk_long_ego"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        if self.observe_steering_angle:
            observation_space_dict["steering_angle"] = gym.spaces.Box(-np.pi, np.pi, (1,), dtype=np.float32)
        if self.observe_relative_orientation:
            observation_space_dict["relative_orientation"] = gym.spaces.Box(-np.pi, np.pi, (1,), dtype=np.float32)
        if self.observe_yaw_rate:
            observation_space_dict["yaw_rate"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        if self.observe_lat_diff_ref_path:
            observation_space_dict["lat_diff_ref_path"] = gym.spaces.Box(-10.0, 10.0, (1,), dtype=np.float32)
        return observation_space_dict

    def observe(self, ego_state, optimal_traj, planner) -> Union[ndarray, Dict]:
        """
        Create ego-related observation for given state in an environment.
        """

        if self.observe_v_ego:
            self.observation_dict["v_ego"] = np.array([ego_state.velocity])

        if self.observe_a_ego:
            self.observation_dict["a_ego"] = np.array([ego_state.acceleration])

        if self.observe_jerk_lat_ego:
            if optimal_traj is not None:
                jerk_lat = optimal_traj.costMap["lateral_jerk"][0]
                self.observation_dict["jerk_lat_ego"] = np.array([jerk_lat])
            else:
                self.observation_dict["jerk_lat_ego"] = np.array([0.0])

        if self.observe_jerk_long_ego:
            if optimal_traj is not None:
                jerk_long = optimal_traj.costMap["longitudinal_jerk"][0]
                self.observation_dict["jerk_long_ego"] = np.array([jerk_long])
            else:
                self.observation_dict["jerk_long_ego"] = np.array([0.0])

        if self.observe_steering_angle:
            if "steering_angle" in ego_state.used_attributes:
                self.observation_dict["steering_angle"] = np.array([ego_state.steering_angle])
            else:
                self.observation_dict["steering_angle"] = np.array([0])

        if self.observe_relative_orientation:
            relative_orientation = self._calculate_relative_orientation(planner, ego_state)
            self.observation_dict["relative_orientation"] = np.array([relative_orientation])

        if self.observe_yaw_rate:
            self.observation_dict["yaw_rate"] = np.array([ego_state.yaw_rate])

        if self.observe_lat_diff_ref_path:
            self.observation_dict["lat_diff_ref_path"] = np.array([planner.x_cl[1][0]])

        return self.observation_dict

    @staticmethod
    def _calculate_relative_orientation(planner, ego_state):
        try:
            ref_position = np.argmin(np.linalg.norm(planner.reference_path - ego_state.position, axis=1))
            orientation_ref_path = calc_orientation_of_line(planner.reference_path[ref_position], planner.reference_path[ref_position + 1])
            return orientation_ref_path - ego_state.orientation
        except:
            return 0.0
