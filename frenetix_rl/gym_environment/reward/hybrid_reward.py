"""Class for hybrid reward"""
import logging
import numpy as np
from collections import OrderedDict

from commonroad.common.solution import PlanningProblemSolution, CostFunction, VehicleModel, VehicleType
from commonroad_dc.costs.evaluation import CostFunctionEvaluator
from cr_scenario_handler.utils.evaluation import create_full_solution_trajectory
from cr_scenario_handler.utils.agent_status import AgentStatus

from frenetix_rl.gym_environment.reward.reward import Reward

LOGGER = logging.getLogger(__name__)


class HybridReward(Reward):
    """Class for hybrid reward"""

    def __init__(self, configs: dict):
        self.reward_configs = configs["reward_configs"]["dense_reward"]
        self.termination_configs = configs["reward_configs"]["sparse_reward"]
        self.last_actions = []  # A list to store the history of actions
        self.max_history = 10   # Maximum history length
        self.test_env = False
        self.config_sim = None
        self.simulation = None
        self.agent = None
        self.reference_cost_weights = None

    def reset(self, test_env, cost_weights):
        self.last_actions = []
        self.test_env = test_env
        self.config_sim = None
        self.simulation = None
        self.agent = None
        self.reference_cost_weights = cost_weights

    def calc_reward(self, observation_dict: OrderedDict, action: np.ndarray, agent, config_sim=None, simulation=None,
                    termination_info=None) -> list:
        """
        Calculate the reward according to the observations

        :param observation_dict: current observations
        :param action: Current action of the environment
        :param agent
        :param config_sim: config information (optional)
        :param simulation: simulation information (optional)
        :param termination_info
        :return: list of all rewards
        """
        self.config_sim = config_sim
        self.simulation = simulation
        self.agent = simulation.agents[0]

        rewards = []

        termination_reward = self.termination_reward(observation_dict, termination_info, agent)
        rewards.append(termination_reward)

        # if self.test_env:
        #     return rewards

        if self.reward_configs["reward_feasible_percentage"] and observation_dict.get("observe_feasible_percentage"):
            rewards.append(self._feasible_percentage_reward(observation_dict))
        else:
            rewards.append(0)

        if self.reward_configs["reward_diff_ref_path"] and observation_dict.get("lat_diff_ref_path"):
            rewards.append(self._diff_ref_path(observation_dict))
        else:
            rewards.append(0)

        # Action inconsistency
        if self.reward_configs["reward_action_inconsistency"]:
            rewards.append(self._action_inconsistency_reward(action))  # Outdated. New method with hold
        else:
            rewards.append(0)

        if self.reward_configs["reward_distance_to_goal_position_advance"] and observation_dict.get("distance_goal_percentage"):
            rewards.append(self._goal_distance_reward(observation_dict))
        else:
            rewards.append(0)

        if self.reward_configs["reward_difference_target_velocity"] and observation_dict.get("difference_desired_velocity_to_goal"):
            rewards.append(self._goal_velocity_distance_reward(observation_dict))
        else:
            rewards.append(0)

        if observation_dict.get("ego_risk"):
            rewards.append(self._ego_risk_reward(observation_dict))
        else:
            rewards.append(0)

        if observation_dict.get("obstacle_risk"):
            rewards.append(self._obst_risk_reward(observation_dict))
        else:
            rewards.append(0)

        # jerk reward
        if self.reward_configs["reward_jerk"]:
            rewards.append(self._jerk_reward(observation_dict))
        else:
            rewards.append(0)

        if self.reward_configs["cost_norm_difference"]:
            rewards.append(self._cost_norm_difference_reward())
        else:
            rewards.append(0)

        return rewards

    def termination_reward(self, observation_dict, termination_info, agent) -> float:
        """Reward for the cause of termination"""

        reward = 0.0

        # Reach goal
        if agent.status == AgentStatus.COMPLETED_SUCCESS:
            LOGGER.debug("GOAL REACHED SUCCESS!")
            reward += self.termination_configs["reward_goal_reached_success"]
        elif agent.status == AgentStatus.COMPLETED_OUT_OF_TIME:
            LOGGER.debug("GOAL REACHED OUT OF TIME!")
            reward += self.termination_configs["reward_goal_reached_out_of_time"]
        elif agent.status == AgentStatus.COMPLETED_FASTER:
            LOGGER.debug("GOAL REACHED FASTER!")
            reward += self.termination_configs["reward_goal_reached_faster"]

        # Collision
        elif agent.status == AgentStatus.COLLISION:
            reward += self.termination_configs["reward_collision"]

        elif agent.status == AgentStatus.ERROR:
            reward += self.termination_configs["reward_not_feasible"]

        # Exceed maximum episode length
        elif agent.status == AgentStatus.TIMELIMIT:
            reward += self.termination_configs["reward_time_out"]

        elif agent.status == AgentStatus.MAX_S_POSITION:
            reward += self.termination_configs["reward_max_s_position"]

        # if scenario is finished, calculate cost of entire scenario
        if reward != 0 and self.termination_configs["rate_scenario_solution"] and observation_dict["is_goal_reached"][0]:
            cost_function = CostFunction(self.termination_configs["scenario_cost_function"])
            vehicle_type = VehicleType(self.config_sim.vehicle.cr_vehicle_id)
            ego_solution_trajectory = create_full_solution_trajectory(self.config_sim, self.agent.record_state_list)
            pps = PlanningProblemSolution(planning_problem_id=self.agent.planning_problem.planning_problem_id,
                                          vehicle_type=vehicle_type,
                                          vehicle_model=VehicleModel.KS,
                                          cost_function=cost_function,
                                          trajectory=ego_solution_trajectory)
            cost_evaluator = CostFunctionEvaluator(cost_function, vehicle_type)
            total_cost = cost_evaluator.evaluate_pp_solution(self.agent.scenario, self.agent.planning_problem,
                                                             pps.trajectory).total_costs
            reward -= total_cost

        return reward

    def _cost_norm_difference_reward(self):
        cost_difference = 0
        for cost_name, cost in self.agent.planner_interface.planner.cost_weights.items():
            if cost_name != "prediction":
                cost_difference += abs(cost - self.reference_cost_weights[cost_name])
        cost_norm_difference = self.reward_configs["cost_norm_difference"] * cost_difference
        cost_difference_prediction = abs(self.agent.planner_interface.planner.cost_weights["prediction"] - self.reference_cost_weights["prediction"])
        cost_norm_difference += self.reward_configs["cost_norm_difference_prediction"] * cost_difference_prediction
        return cost_norm_difference

    def _feasible_percentage_reward(self, observation_dict):
        return self.reward_configs["reward_feasible_percentage"] * observation_dict["observe_feasible_percentage"][0]

    def _diff_ref_path(self, observation_dict: dict) -> float:
        return self.reward_configs["reward_diff_ref_path"] * abs(observation_dict["lat_diff_ref_path"][0])

    def _action_inconsistency_reward(self, action) -> float:
        """
        Penalty for having a high variance in the actions up to 10 time steps in the past.
        """
        # Add the current action to the history
        self.last_actions.append(action)

        # Only keep the last 10 actions
        if len(self.last_actions) > self.max_history:
            self.last_actions.pop(0)

        # Calculate the mean action over the history for normalization
        mean_action = np.mean(self.last_actions, axis=0)

        # Calculate the difference of each action from the mean action
        diffs = [np.linalg.norm(a - mean_action) for a in self.last_actions]

        # The negative reward is the sum of the differences
        inconsistency_penalty = sum(diffs)

        # Normalize the penalty by the number of time steps considered
        normalized_penalty = inconsistency_penalty / len(self.last_actions)

        return self.reward_configs['reward_action_inconsistency'] * normalized_penalty

    def _goal_distance_reward(self, observation_dict: dict) -> float:
        if observation_dict["distance_goal_percentage"][0] == 1:
            return 0
        else:
            return self.reward_configs["reward_distance_to_goal_position_advance"] * observation_dict["distance_goal_percentage"][0]

    def _goal_velocity_distance_reward(self, observation_dict):
        return self.reward_configs["reward_difference_target_velocity"] * abs(observation_dict["difference_desired_velocity_to_goal"][0])

    def _ego_risk_reward(self, observation_dict: dict) -> float:
        return self.reward_configs["reward_risk_ego"] * observation_dict["ego_risk"][0]

    def _obst_risk_reward(self, observation_dict: dict) -> float:
        return self.reward_configs["reward_risk_obst"] * observation_dict["obstacle_risk"][0]

    def _jerk_reward(self, observation_dict: dict) -> float:
        jerk = observation_dict["jerk_lat_ego"][0] + observation_dict["jerk_long_ego"][0]
        return self.reward_configs["reward_jerk"] * jerk
