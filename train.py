__author__ = "Rainer Trauth, Alexander Hobmeier"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Release"

import os
from stable_baselines3.common.callbacks import CallbackList
from sb3_contrib import RecurrentPPO
import frenetix_rl.utils.helper_functions as hf
from frenetix_rl.gym_environment.paths import PATH_PARAMS
from frenetix_rl.gym_environment.utils.callback_helpers import (
    TensorboardCallback,
    create_checkpoint_callback
)
from stable_baselines3.common.callbacks import EvalCallback


def main():

    mod_path = os.path.dirname(os.path.abspath(__file__))

    ##############################################
    # Read Environment and Scenario Configurations
    ##############################################
    env_configs_file = PATH_PARAMS["configs"]
    env_configs = hf.load_environment_configs(env_configs_file)
    training_configs = env_configs["training_configs"]

    scenario_dir = os.path.join(mod_path, "scenarios")
    test_scenario_dir = os.path.join(mod_path, "scenarios_validation")

    if not os.path.isdir(scenario_dir) or not os.path.isdir(test_scenario_dir):
        raise FileNotFoundError("Scenarios or scenarios_validation folder does not exist!")

    if not os.listdir(scenario_dir) or not os.listdir(test_scenario_dir):
        raise FileNotFoundError("Scenarios or scenarios_validation folder does not contain any scenarios!")

    ###############
    # Save Settings
    ###############
    logs_path = PATH_PARAMS["logs"]
    logs_tensorboard = PATH_PARAMS["logs_tensorboard"]
    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(logs_tensorboard, exist_ok=True)
    hf.save_environment_configs(os.path.join(logs_path, "environment_configurations.yml"), env_configs)

    ######################
    # Read Hyperparameters
    ######################
    hyperparams_file = PATH_PARAMS["hyperparams"]
    hyperparams = hf.load_hyperparameters(hyperparams_file)
    hf.save_hyperparameters(os.path.join(logs_path, "model_hyperparameters.yml"), hyperparams)

    ########################
    # Create the Environment
    ########################
    num_envs = training_configs["num_envs"]

    training_env = hf.create_vectorized_training_environments(
        scenario_dir=scenario_dir,
        env_configs=env_configs,
        num_envs=num_envs
    )

    testing_env = hf.create_vectorized_testing_environments(
        scenario_dir=test_scenario_dir,
        env_configs=env_configs,
        num_envs=num_envs,
    )

    ##################
    # Create Callbacks
    ##################
    eval_freq = training_configs["eval_freq"]
    n_eval_episodes = training_configs["n_eval_episodes"]
    best_model_save_path = os.path.join(logs_path, "best_model")
    intermediate_model_path = os.path.join(logs_path, "intermediate_model")
    intermediate_model_save_feq = training_configs["intermediate_model_save_feq"]

    eval_callback = EvalCallback(testing_env,
                                 log_path=os.path.join(logs_path, "eval"),
                                 eval_freq=eval_freq,
                                 n_eval_episodes=n_eval_episodes,
                                 callback_on_new_best=None,
                                 best_model_save_path=best_model_save_path)

    # Save the model periodically
    checkpoint_callback = create_checkpoint_callback(intermediate_model_save_feq, intermediate_model_path)
    # Create a list of callbacks
    callbacks = CallbackList([TensorboardCallback(env_configs), checkpoint_callback, eval_callback])

    ################
    # Start Training
    ################
    model = RecurrentPPO("MlpLstmPolicy", env=training_env, **hyperparams)
    total_timesteps = training_configs["total_timesteps"]
    model.learn(total_timesteps=total_timesteps, callback=callbacks, tb_log_name='PPO', progress_bar=True)


if __name__ == '__main__':
    main()
