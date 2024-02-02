__author__ = "Rainer Trauth, Alexander Hobmeier"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Release"

"""Module containing the Termination class"""
from cr_scenario_handler.utils.agent_status import AgentStatus


class Termination:
    """Class for detecting if the scenario should be terminated"""

    def __init__(self, config: dict):
        """
        :param config: Configuration of the environment
        """
        self.termination_configs = config["termination_configs"]

    def reset(self):
        pass

    def is_terminated(self, agent, observation: dict) -> (bool, str, dict):
        """
        Detect if the scenario should be terminated
        :param agent
        :param observation: Current observation of the environment
        :return: Tuple of (terminated: bool, reason: str, termination_info: dict)
        """

        done = False
        termination_info = {
            "is_goal_reached_success": 0,
            "is_goal_reached_out_of_time": 0,
            "is_goal_reached_faster": 0,
            "is_collision": 0,
            "is_time_out": 0,
            "no_feasible_solution": 0,
            "max_s_position": 0
        }

        termination_reason = None

        if agent.status == AgentStatus.COLLISION:  # Collision with others
            termination_info["is_collision"] = 1
            if self.termination_configs["terminate_on_collision"]:
                done = True
                termination_reason = "is_collision"

        elif agent.status == AgentStatus.ERROR:
            termination_info["no_feasible_solution"] = 1
            if self.termination_configs["terminate_on_infeasibility"]:
                done = True
                termination_reason = "not_feasible"

        elif agent.status == AgentStatus.TIMELIMIT:  # Max simulation time step is reached
            termination_info["is_time_out"] = 1
            if self.termination_configs["terminate_on_time_out"]:
                done = True
                termination_reason = "is_time_out"

        elif agent.status == AgentStatus.MAX_S_POSITION:  # Max simulation time step is reached
            termination_info["max_s_position"] = 1
            if self.termination_configs["terminate_on_max_s_position"]:
                done = True
                termination_reason = "max_s_position"

        elif agent.status == AgentStatus.COMPLETED_FASTER:
            termination_info["is_goal_reached_faster"] = 1
            if self.termination_configs["terminate_on_goal_reached"]:
                done = True
                termination_reason = "is_goal_reached_faster"

        elif agent.status == AgentStatus.COMPLETED_OUT_OF_TIME:
            termination_info["is_goal_reached_out_of_time"] = 1
            if self.termination_configs["terminate_on_goal_reached"]:
                done = True
                termination_reason = "is_goal_reached_out_of_time"

        elif agent.status == AgentStatus.COMPLETED_SUCCESS:
            termination_info["is_goal_reached_success"] = 1
            if self.termination_configs["terminate_on_goal_reached"]:
                done = True
                termination_reason = "is_goal_reached_success"

        return done, termination_reason, termination_info
