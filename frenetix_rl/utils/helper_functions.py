import os
import sys
import yaml
import logging
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from frenetix_rl.gym_environment.environment.agent_env import AgentEnv

LOGGER = logging.getLogger(__name__)


def count_files_in_directory(directory):
    return sum(os.path.isfile(os.path.join(directory, f)) for f in os.listdir(directory))


def load_environment_configs(config_file_path):
    with open(config_file_path, "r") as config_file:
        env_configs = yaml.safe_load(config_file)["env_configs"]
    return env_configs


def save_environment_configs(config_file_path, env_configs):
    with open(config_file_path, "w") as config_file:
        yaml.dump({"env_configs": env_configs}, config_file)


def load_hyperparameters(hyperparam_file_path):
    with open(hyperparam_file_path, "r") as hyperparam_file:
        hyperparams = yaml.safe_load(hyperparam_file)["commonroad-v1"]
    return hyperparams


def save_hyperparameters(hyperparam_file_path, hyperparams):
    with open(hyperparam_file_path, "w") as hyperparam_file:
        yaml.dump({"commonroad-v1": hyperparams}, hyperparam_file)


def register_commonroad_env(mod_path):

    # Add the module path to the system path
    sys.path.append(mod_path)

    gym.envs.registration.register(
        id='commonroad-v1',
        entry_point='frenetix_rl.gym_environment.environment.agent_env:AgentEnv'
    )


def create_vectorized_training_environments(scenario_dir, env_configs, num_envs):
    # load scenario paths
    scenario_paths = []
    for r, d, f in os.walk(scenario_dir):
        for file in f:
            scenario_paths.append(os.path.join(r, file))
    # create environment callables
    training_env = lambda: gym.make("commonroad-v1", scenario_paths=scenario_paths, env_configs=env_configs)
    return SubprocVecEnv([training_env for _ in range(num_envs)])


def create_vectorized_testing_environments(scenario_dir, env_configs, num_envs):
    # load testing sceanrio paths
    scenario_paths = []
    for r, d, f in os.walk(scenario_dir):
        for file in f:
            scenario_paths.append(os.path.join(r, file))
    # split scenarios for all environments and create the environments
    environments = []
    if len(scenario_paths) > env_configs["training_configs"]["n_eval_episodes"]:
       LOGGER.warning("There are less than n_eval_episodes scenarios in the testing scenario folder")
    if len(scenario_paths) < num_envs:
        for i in range(len(scenario_paths)):
            testing_env = AgentEnv(scenario_paths=[scenario_paths[i]], env_configs=env_configs, test_env=True,
                                            pick_random_scenario=False, plot_agents=False)
            environments.append(testing_env)
    else:
        scenarios_per_env = int(len(scenario_paths) / num_envs)
        if scenarios_per_env * num_envs  != len(scenario_paths):
            LOGGER.warning("The scenarios are not divisible by the number of environments, some scenarios will be cut off")
        for i in range(num_envs):
            testing_scenarios_i = scenario_paths[i * scenarios_per_env:(i + 1) * scenarios_per_env]
            testing_env = AgentEnv(scenario_paths=testing_scenarios_i, env_configs=env_configs, test_env=True,
                                            pick_random_scenario=False, plot_agents=False)
            environments.append(testing_env)
    callables = []
    for i in range(len(environments)):
        callables.append(lambda i_env=i: environments[i_env])
    return SubprocVecEnv(callables)
