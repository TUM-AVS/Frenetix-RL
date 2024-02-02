__author__ = "Rainer Trauth, Alexander Hobmeier"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Release"

from collections import OrderedDict
from typing import Dict, Union
import numpy as np
from gymnasium.spaces import Box

from frenetix_rl.gym_environment.observation import Observation


class SurroundingObservation(Observation):

    def __init__(self, configs: Dict, configs_name: str = "surrounding_configs"):
        """
        :param configs: dictionary to store all observation configurations
        :param configs_name: key of configs dictionary corresponding to this observation
        """

        surrounding_configs = configs["observation_configs"][configs_name]
        self.observe_adjacent_lanes = surrounding_configs["observe_adjacent_lanes"]
        self.observe_obstacles = surrounding_configs["observe_obstacles"]
        self.obstacle_amount = surrounding_configs["observe_obstacle_amount"]

        self.observation_dict = OrderedDict()

    def build_observation_space(self) -> OrderedDict:
        observation_space_dict = OrderedDict()
        if self.observe_adjacent_lanes:
            observation_space_dict["adj_left"] = Box(0, 1, (1,), dtype=np.int8)
            observation_space_dict["adj_left_same_direction"] = Box(-1, 1, (1,), dtype=np.int8)
            observation_space_dict["adj_right"] = Box(0, 1, (1,), dtype=np.int8)
            observation_space_dict["adj_right_same_direction"] = Box(-1, 1, (1,), dtype=np.int8)

        if self.observe_obstacles:
            for i in range(self.obstacle_amount):
                observation_space_dict["obstacles "+str(i)] = Box(-np.inf, np.inf, (7, ), dtype=np.float32)

        return observation_space_dict

    def observe(self, scenario, ego_state, optimal_traj) -> Union[np.array, Dict]:
        """
        Observation of the surrounding obstacles.
        """
        lln = scenario.lanelet_network
        position = ego_state.position
        position_transformed = [[position[0], position[1]]]
        ll_index = lln.find_lanelet_by_position(position_transformed)[0]
        if not ll_index:
            return self.observation_dict

        if self.observe_adjacent_lanes:
            adj_left, adj_left_same_direction, adj_right, adj_right_same_direction = self.get_lanelet_info(lln, ll_index)
            self.observation_dict["adj_left"] = np.array([adj_left])
            self.observation_dict["adj_left_same_direction"] = np.array([adj_left_same_direction])
            self.observation_dict["adj_right"] = np.array([adj_right])
            self.observation_dict["adj_right_same_direction"] = np.array([adj_right_same_direction])

        if self.observe_obstacles:
            data = self.get_obstacle_info(scenario, ego_state, ll_index, optimal_traj)
            for i in range(self.obstacle_amount):
                obstacle_data = np.zeros(5)
                if i < len(data):
                    obstacle_data = data[i]
                self.observation_dict["obstacles "+str(i)] = obstacle_data

        return self.observation_dict

    def get_lanelet_info(self, lln, ll_index):
        lanelet = lln.find_lanelet_by_id(ll_index[0])
        adj_left = True if lanelet.adj_left else False
        adj_left_same_direction = -1 if lanelet.adj_left_same_direction is None else lanelet.adj_left_same_direction
        adj_right = True if lanelet.adj_right else False
        adj_right_same_direction = -1 if lanelet.adj_right_same_direction is None else lanelet.adj_right_same_direction

        return adj_left, adj_left_same_direction, adj_right, adj_right_same_direction

    def get_obstacle_info(self, scenario, ego_state, ego_ll_index, optimal_traj):
        list_obstacle_data = []
        # iterate over all obstacles
        for obstacle in scenario.dynamic_obstacles:
            obst_traj = obstacle.prediction.trajectory
            if ego_state.time_step < obstacle.prediction.initial_time_step or ego_state.time_step > obstacle.prediction.final_time_step:
                continue
            obstacle_state = obst_traj.state_list[ego_state.time_step - obstacle.prediction.initial_time_step]
            obstacle_distance = np.linalg.norm(ego_state.position-obstacle_state.position)
            obstacle_orientation_diff = normalize_angle(obstacle_state.orientation - ego_state.orientation)

            # get lanelet data
            obstacle_pos_transformed = [[obstacle_state.position[0], obstacle_state.position[1]]]
            obstacle_lanelet_index = scenario.lanelet_network.find_lanelet_by_position(obstacle_pos_transformed)[0]
            obstacle_same_lanelet = obstacle_lanelet_index == ego_ll_index

            # obstacle_in_front_of_ego = -1
            # if optimal_traj is not None:
            #     # get positions
            #     current_pos = np.array([optimal_traj.cartesian.x[0], optimal_traj.cartesian.y[0]])
            #     next_pos = np.array([optimal_traj.cartesian.x[1], optimal_traj.cartesian.y[1]])
            #     obstacle_pos = obstacle_state.position
            #     # get position vectors
            #     direction_travel = next_pos - current_pos
            #     direction_obstacle = obstacle_pos - current_pos
            #     # calculate angle
            #     cosine_angle = np.dot(direction_travel, direction_obstacle) / (np.linalg.norm(direction_travel) * np.linalg.norm(direction_obstacle))
            #     cosine_angle = np.clip(cosine_angle, -1, 1)
            #     angle = np.arccos(cosine_angle)
            #     # determine if obstacle is in front of ego
            #     obstacle_in_front_of_ego = angle < np.pi/2

            # create observation vector
            list_obstacle_data.append(np.array([obstacle_state.position[0], obstacle_state.position[1], obstacle_state.orientation,
                                                obstacle_state.velocity, obstacle_distance,
                                                obstacle_orientation_diff, obstacle_same_lanelet]))

        list_obstacle_data.sort(key=lambda x: x[0])
        return list_obstacle_data


def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle
