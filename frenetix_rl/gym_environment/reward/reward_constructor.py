"""File containing a reward function generator (Can't be incorporated into other modules because of cyclic import
problems """
from frenetix_rl.gym_environment.reward.hybrid_reward import HybridReward
from frenetix_rl.gym_environment.reward.reward import Reward

reward_type_to_class = {
    "hybrid_reward": HybridReward,
}


def make_reward(configs: dict) -> Reward:
    """
    Initializes the reward class according to the env_configurations

    :param configs: The configuration of the environment
    :return: Reward class, either hybrid, sparse or dense
    """

    reward_type = configs["reward_type"]

    if reward_type in reward_type_to_class:
        return reward_type_to_class[reward_type](configs)
    else:
        raise ValueError(f"Illegal reward type: {reward_type}!")
