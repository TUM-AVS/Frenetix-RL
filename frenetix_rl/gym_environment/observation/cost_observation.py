__author__ = "Rainer Trauth, Alexander Hobmeier"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Release"

from collections import OrderedDict
from typing import Dict, Union
import gymnasium as gym
import numpy as np

from frenetix_rl.gym_environment.observation import Observation


class CostObservation(Observation):
    """observation Class to observe the cost functions"""

    def __init__(self, configs: Dict, configs_name: str = "cost_configs"):
        """
        :param configs: dictionary to store all observation configurations
        :param configs_name: key of configs dictionary corresponding to this observation
        """

        cost_function_configs = configs["observation_configs"][configs_name]
        self.cost_terms = configs["action_configs"]["cost_terms"]
        self.n_costs = len(self.cost_terms)

        self.observe_cost_optimal_traj = cost_function_configs["observe_cost_optimal_traj"]
        self.observe_cost_mean = cost_function_configs["observe_cost_mean"]
        self.observe_cost_variance = cost_function_configs["observe_cost_variance"]
        self.observe_cost_predictions = cost_function_configs["observe_cost_predictions"]
        # self.observe_cost_distances = cost_function_configs["observe_cost_distances"]
        self.observe_current_weights = cost_function_configs["observe_current_weights"]
        self.observation_dict = OrderedDict()
        self.max_t = None

    def build_observation_space(self) -> OrderedDict:
        """ build observations space for the cost function observation """
        observation_space_dict = OrderedDict()

        if self.observe_cost_optimal_traj:
            observation_space_dict["cost_optimal_traj"] = gym.spaces.Box(-np.inf, np.inf, (self.n_costs,), dtype=np.float32)
        if self.observe_cost_mean:
            observation_space_dict["cost_mean"] = gym.spaces.Box(-np.inf, np.inf, (self.n_costs,), dtype=np.float32)
        if self.observe_cost_variance:
            observation_space_dict["cost_variance"] = gym.spaces.Box(-np.inf, np.inf, (self.n_costs,), dtype=np.float32)
        if self.observe_cost_predictions:
            observation_space_dict["cost_predictions"] = gym.spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32)
        if self.observe_current_weights:
            observation_space_dict["current_weights"] = gym.spaces.Box(0.0, 400.0, (len(self.cost_terms) + 1,), dtype=np.float32)
        # if self.observe_cost_distances:
        #     observation_space_dict["cost_distances"] = gym.spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32)

        return observation_space_dict

    def observe(self, agent, optimal_traj) -> Union[np.array, Dict]:
        # only compute mean and variance if trajectory bundle exists
        if agent.all_trajectories:
            cost_mean, cost_variance = self.calculate_cost_function_observation(agent.all_trajectories)
        else:
            cost_mean = np.zeros((self.n_costs,), dtype=np.float32)
            cost_variance = np.zeros((self.n_costs,), dtype=np.float32)
        # get cost of optimal trajectory
        if optimal_traj:
            cost_optimal_traj = self.get_cost_optimal_traj(optimal_traj)
        else:
            cost_optimal_traj = np.zeros((self.n_costs,), dtype=np.float32)

        if agent.all_trajectories:
            cost_predictions = self.calculate_costs_predictions_distances(agent)
        else:
            cost_predictions = np.zeros((9,), dtype=np.float32)
            # cost_distances = np.zeros((9,), dtype=np.float32)

        current_weights = np.zeros(len(self.cost_terms) + 1)
        i = 0
        for costs in agent.planner_interface.planner.cost_weights:
            if costs in self.cost_terms or costs == "prediction":
                current_weights[i] = agent.planner_interface.planner.cost_weights[costs]
                i += 1

        if self.observe_cost_optimal_traj:
            self.observation_dict["cost_optimal_traj"] = cost_optimal_traj
        if self.observe_cost_mean:
            self.observation_dict["cost_mean"] = cost_mean
        if self.observe_cost_variance:
            self.observation_dict["cost_variance"] = cost_variance
        if self.observe_cost_predictions:
            self.observation_dict["cost_predictions"] = cost_predictions
        if self.observe_current_weights:
            self.observation_dict["current_weights"] = current_weights
        # if self.observe_cost_distances:
        #     self.observation_dict["cost_distances"] = cost_distances

        return self.observation_dict

    def calculate_cost_function_observation(self, trajectories):

        cost_lists = np.zeros((self.n_costs, len(trajectories)))

        # save the cost list of each trajectory in a column of the array
        for i, trajectory in enumerate(trajectories):
            for j, cost_term in enumerate(self.cost_terms):
                cost_value = trajectory.costMap.get(cost_term, [0.0])[0]  # Use get method with default [0.0]
                if not np.isfinite(cost_value):  # Check for inf or nan
                    cost_value = 0.0
                cost_lists[j, i] = cost_value

        # compute mean of each row of the array
        cost_mean = np.mean(cost_lists, axis=1)
        # compute variance of each row of the array
        cost_variance = np.var(cost_lists, axis=1)

        # Handle division by zero in normalization by using max, if norm is 0 then the vector is zero
        norm_cost_mean = np.linalg.norm(cost_mean)
        normalized_cost_mean = cost_mean / norm_cost_mean if norm_cost_mean != 0 else cost_mean

        norm_cost_variance = np.linalg.norm(cost_variance)
        normalized_cost_variance = cost_variance / norm_cost_variance if norm_cost_variance != 0 else cost_variance

        return normalized_cost_mean, normalized_cost_variance

    def get_cost_optimal_traj(self, optimal_traj):
        cost_list = np.zeros(self.n_costs)

        for i, cost_term in enumerate(self.cost_terms):
            if cost_term in optimal_traj.costMap.keys():
                cost_list[i] = optimal_traj.costMap[cost_term][0]

        return cost_list

    def calculate_costs_predictions_distances(self, agent):

        # Retrieve all trajectories and determine the max_t if not set
        trajectories = agent.all_trajectories
        if not trajectories:
            return np.zeros(9)

        if not self.max_t:
            self.max_t = max(traj.sampling_parameters[1] for traj in trajectories)

        # Filter trajectories with max_t
        max_t_trajectories = [traj for traj in trajectories if traj.sampling_parameters[1] == self.max_t]
        if not max_t_trajectories:
            return np.zeros(9)

        # Create sets for unique velocity and distance values and sort them
        unique_v_values = sorted(set(traj.sampling_parameters[5] for traj in max_t_trajectories) - {agent.planner_interface.planner.x_cl[0][1]})
        unique_d_values = sorted(
            set(traj.sampling_parameters[10] for traj in max_t_trajectories) - {agent.planner_interface.planner.x_cl[1][0]})

        # Define grid positions based on unique values
        grid_positions = [(v, d) for v in
                          [unique_v_values[0], unique_v_values[len(unique_v_values) // 2], unique_v_values[-1]]
                          for d in
                          [unique_d_values[0], unique_d_values[len(unique_d_values) // 2], unique_d_values[-1]]]

        # Initialize arrays for cost predictions and distances
        cost_predictions = np.zeros(9)
        # cost_distances = np.zeros(9)

        # Fill in the arrays with the corresponding cost predictions and distances
        for i, (v_value, d_value) in enumerate(grid_positions):
            # Find the trajectory that matches the velocity and distance values
            trajectory = next((traj for traj in max_t_trajectories if traj.sampling_parameters[5] == v_value
                               and traj.sampling_parameters[10] == d_value), None)
            if trajectory:  # If a matching trajectory is found, use its values
                cost_predictions[i] = trajectory.costMap["prediction"][0]
                # cost_distances[i] = trajectory.costMap["distance_to_obstacles"][0]

        return cost_predictions


