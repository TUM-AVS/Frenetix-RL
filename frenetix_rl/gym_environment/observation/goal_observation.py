from collections import OrderedDict
from typing import Dict

import gymnasium as gym
import numpy as np
from commonroad.planning.goal import GoalRegion
from cr_scenario_handler.utils.agent_status import AgentStatus
from frenetix_rl.gym_environment.observation.observation import Observation


class GoalObservation(Observation):
    """
    This class contains all helper methods and the main observation method for observations related to the goal

    :param configs: the configuration specification
    """

    def __init__(self, configs: Dict, config_name: str = "goal_configs"):
        # Read configs
        configs = configs["observation_configs"][config_name]
        self.observe_distance_goal = configs.get("observe_distance_goal")
        self.observe_remaining_steps = configs.get("observe_remaining_steps")
        self.observe_is_goal_reached = configs.get("observe_is_goal_reached")
        self.observe_is_goal_reached_position = configs.get("observe_is_goal_reached_position")
        self.observe_is_goal_reached_velocity = configs.get("observe_is_goal_reached_velocity")
        self.observe_is_time_out = configs.get("observe_is_time_out")
        self.observe_difference_desired_velocity_to_goal = configs.get("observe_difference_desired_velocity_to_goal")

        self.goal_region = None
        self.episode_length = None
        self.goal_state = None
        self.goal_has_position_attribute = None

        # goal positions
        self.first_goal_position = None
        self.initial_ego_position = None
        self.scenario_distance = None
        self.last_distance_goal_long = None

        # location for storing the past observations
        self.observation_history_dict: dict = dict()

    def reset(self, planning_problem, agent):
        self.goal_region: GoalRegion = planning_problem.goal
        self.episode_length = max(s.time_step.end for s in self.goal_region.state_list)
        self.goal_state = self.goal_region.state_list[0]

        self.goal_has_position_attribute = "position" in self.goal_state.attributes

        if self.goal_has_position_attribute:
            first_pos_in_goal = next(
                (point for point in agent.reference_path if self.goal_state.position.contains_point(point)), None)

            assert first_pos_in_goal is not None, "No Point of the Reference Path is in the Goal Area"

            self.first_goal_position = (agent.planner_interface.planner.coordinate_system.ccosy.
                                        convert_to_curvilinear_coords(first_pos_in_goal[0], first_pos_in_goal[1]))
            initial_pos = planning_problem.initial_state.position
            self.initial_ego_position = (agent.planner_interface.planner.coordinate_system.ccosy.
                                         convert_to_curvilinear_coords(initial_pos[0], initial_pos[1]))
            self.scenario_distance = self.first_goal_position[0] - self.initial_ego_position[0]
            assert self.scenario_distance > 0.0, "Goal Area is behind ego vehicle start position!"
            self.last_distance_goal_long = self.scenario_distance

    def build_observation_space(self) -> OrderedDict:
        observation_space_dict = OrderedDict()

        if self.observe_distance_goal:
            observation_space_dict["distance_goal_long"] = gym.spaces.Box(0, np.inf, (1,), dtype=np.float32)
            observation_space_dict["distance_goal_percentage"] = gym.spaces.Box(0, 1, (1,), dtype=np.float32)

        if self.observe_remaining_steps:
            observation_space_dict["remaining_steps"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.int16)

        if self.observe_is_goal_reached:
            observation_space_dict["is_goal_reached"] = gym.spaces.Box(0, 1, (1,), dtype=np.int8)

        if self.observe_is_goal_reached_position:
            observation_space_dict["is_goal_reached_position"] = gym.spaces.Box(0, 1, (1,), dtype=np.int8)

        if self.observe_is_goal_reached_velocity:
            observation_space_dict["is_goal_reached_velocity"] = gym.spaces.Box(0, 1, (1,), dtype=np.int8)

        if self.observe_is_time_out:
            observation_space_dict["is_time_out"] = gym.spaces.Box(0, 1, (1,), dtype=np.int8)

        if self.observe_difference_desired_velocity_to_goal:
            observation_space_dict["difference_desired_velocity_to_goal"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)

        return observation_space_dict

    def observe(self, planner, ego_state, agent, status):
        observation_dict = {}

        distance_goal_long, distance_goal_percentage = self.get_distance_to_goal(planner)

        if self.observe_distance_goal:
            observation_dict["distance_goal_long"] = np.array([distance_goal_long])
            observation_dict["distance_goal_percentage"] = np.array([distance_goal_percentage])

        # observe the time to the goal state
        if self.observe_remaining_steps:
            remaining_time = self.get_remaining_time(ego_state)
            observation_dict["remaining_steps"] = np.array([remaining_time])

        if self.observe_is_goal_reached:
            if agent.status == (AgentStatus.COMPLETED_SUCCESS or
                                AgentStatus.COMPLETED_FASTER or
                                AgentStatus.COMPLETED_OUT_OF_TIME):
                goal_reached = True
            else:
                goal_reached = False
            observation_dict["is_goal_reached"] = np.array([goal_reached])

        if self.observe_is_goal_reached_position:
            if agent.agent_state.goal_checker_status and "position" in agent.agent_state.goal_checker_status:
                observation_dict["is_goal_reached_position"] = np.array([agent.agent_state.goal_checker_status["position"]])
            else:
                observation_dict["is_goal_reached_position"] = np.array([0])

        if self.observe_is_goal_reached_velocity:
            if agent.agent_state.goal_checker_status and "velocity" in agent.agent_state.goal_checker_status:
                observation_dict["is_goal_reached_velocity"] = np.array([agent.agent_state.goal_checker_status["position"]])
            else:
                observation_dict["is_goal_reached_velocity"] = np.array([0])

        if self.observe_is_time_out:
            if agent.status == AgentStatus.TIMELIMIT:
                observation_dict["is_time_out"] = np.array([1])
            else:
                observation_dict["is_time_out"] = np.array([0])

        if self.observe_difference_desired_velocity_to_goal:
            if agent.planner_interface.desired_velocity:
                difference_desired_velocity_to_goal = ego_state.velocity - agent.planner_interface.desired_velocity
                observation_dict["difference_desired_velocity_to_goal"] = np.array([difference_desired_velocity_to_goal])
            else:
                observation_dict["difference_desired_velocity_to_goal"] = np.array([0.0])

        return observation_dict

    def get_distance_to_goal(self, planner):
        if not self.goal_has_position_attribute:
            return 0, 0

        distance_goal_long = self.first_goal_position[0] - planner.x_cl[0][0]
        if distance_goal_long < 0:
            distance_goal_long = 0

        # Calculate the percentage of the total distance that has been covered
        progress_percentage = (self.scenario_distance - distance_goal_long) / self.scenario_distance \
                               if self.scenario_distance else 0
        progress_percentage = max(0, min(progress_percentage, 1))  # Clip the value between 0 and 1

        # Calculate the progress change since the last step
        self.last_distance_goal_long = distance_goal_long

        return distance_goal_long, progress_percentage

    def get_remaining_time(self, ego_state):
        return self.episode_length - ego_state.time_step

