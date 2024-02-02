from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, configs, verbose=0):
        super().__init__(verbose)
        self.configs = configs
        self.sum_termination_reward_overall = 0.0
        self.sum_dense_reward_overall = 0.0

        self.sum_reward_lat_diff_ref_path = 0.0
        self.sum_reward_distance_goal_long_advance = 0.0
        self.sum_reward_difference_goal_velocity = 0.0
        self.sum_reward_ego_risk = 0.0
        self.sum_reward_obstacle_risk = 0.0
        self.sum_reward_cost_norm_difference = 0.0

    def _on_step(self) -> bool:
        # Log actions to tensorboard
        if "chosen_action" not in self.locals["infos"][0].keys():
            return True
        actions = self.locals["infos"][0]["chosen_action"]
        self.logger.record("action/Prediction", actions[0])

        for i, cost_term in enumerate(self.configs["action_configs"]["cost_terms"]):
            self.logger.record("action/cost_"+str(cost_term), actions[1+i])

        # Log rewards to tensorboard
        rewards = self.locals["infos"][0]["reward_list"]
        self.logger.record("reward/reward_termination", rewards[0])
        # self.logger.record("reward/reward_feasible_percentage", rewards[1])
        self.logger.record("reward/reward_lat_diff_ref_path", rewards[2])
        # self.logger.record("reward/reward_action_inconsistency", rewards[3])
        self.logger.record("reward/reward_distance_goal_long_advance", rewards[4])
        self.logger.record("reward/reward_difference_goal_velocity", rewards[5])
        self.logger.record("reward/reward_ego_risk", rewards[6])
        self.logger.record("reward/reward_obstacle_risk", rewards[7])
        # self.logger.record("reward/reward_jerk", rewards[8])
        self.logger.record("reward/reward_cost_norm_difference", rewards[9])
        # self.logger.record("reward/const_reward_progress", rewards[9])

        self.sum_dense_reward_overall += sum(rewards[1:])
        self.logger.record("reward/sum_dense_reward", self.sum_dense_reward_overall)

        self.sum_termination_reward_overall += rewards[0]
        self.logger.record("reward/sum_termination_reward", self.sum_termination_reward_overall)

        # Sum Rewards
        self.sum_reward_lat_diff_ref_path += rewards[2]
        self.logger.record("reward/sum_reward_lat_diff_ref_path", self.sum_reward_lat_diff_ref_path)

        self.sum_reward_distance_goal_long_advance += rewards[4]
        self.logger.record("reward/sum_reward_distance_goal_long_advance", self.sum_reward_distance_goal_long_advance)

        self.sum_reward_difference_goal_velocity += rewards[5]
        self.logger.record("reward/sum_reward_difference_goal_velocity", self.sum_reward_difference_goal_velocity)

        self.sum_reward_ego_risk += rewards[6]
        self.logger.record("reward/sum_reward_ego_risk", self.sum_reward_ego_risk)

        self.sum_reward_obstacle_risk += rewards[7]
        self.logger.record("reward/sum_reward_obstacle_risk", self.sum_reward_obstacle_risk)

        self.sum_reward_cost_norm_difference += rewards[9]
        self.logger.record("reward/sum_reward_cost_norm_difference", self.sum_reward_cost_norm_difference)

        # Log observations
        obs = self.locals["infos"][0]["observation_dict"]
        for obs_key in obs:
            o = obs[obs_key]
            for i in range(len(o)):
                self.logger.record("obs/" + obs_key + "_" + str(i), o[i])

        # dump the logs every 1000 steps
        if self.num_timesteps % 1000 == 0:
            self.logger.dump(self.num_timesteps)
        return True


def create_checkpoint_callback(save_fequency, intermediate_model_path):
    return CheckpointCallback(save_freq=save_fequency, save_path=intermediate_model_path, verbose=1)


