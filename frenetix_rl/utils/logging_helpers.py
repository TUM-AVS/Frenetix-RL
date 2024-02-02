import os


class AgentLogger:
    # ----------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    def __init__(self, path_logs: str, cost_terms: list) -> None:
        """"""

        self.log_file_path = "agent_logs.csv"
        self.cost_terms = cost_terms
        self.header = None

        self.agent_log_path = os.path.join(path_logs, self.log_file_path)

        self.__trajectories_log_path = None

        # Create directories
        if not os.path.exists(path_logs):
            os.makedirs(path_logs)

        self.set_logging_header()

    # ----------------------------------------------------------------------------------------------------------
    # CLASS METHODS --------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    def set_logging_header(self):

        cost_terms_header = ""
        for cost in self.cost_terms :
            cost_terms_header += str(cost) + "_action;"

        self.header = (
            "timestep;"
            "prediction_action;"
            + cost_terms_header +
            "termination_reward;"
            "reward_feasible_percentage;"
            "reward_diff_ref_path;"
            "reward_action_inconsistency;"
            "reward_distance_to_goal_position_advance;"
            "reward_difference_goal_velocity;"
            "ego_risk_reward;"
            "obstacle_risk_reward;"
            "reward_jerk;"
        )

        with open(self.agent_log_path, "w+") as fh:
            fh.write(self.header)

    def log_progress(self, timestep, action, rewards):

        new_line = "\n" + str(timestep)

        for i in range(0, len(action)):
            new_line += ";" + str(action[i])

        for i in range(0, len(rewards)):
            new_line += ";" + str(rewards[i])

        with open(self.agent_log_path, "a") as fh:
            fh.write(new_line)
