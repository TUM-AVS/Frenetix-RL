__author__ = "Rainer Trauth, Alexander Hobmeier"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Release"

import os
from os import listdir
from os.path import isfile, join

from sb3_contrib import RecurrentPPO
from frenetix_rl.gym_environment.paths import PATH_PARAMS
from frenetix_rl.gym_environment.environment.agent_env import AgentEnv
from frenetix_rl.utils.helper_functions import load_environment_configs
from frenetix_rl.evaluation.agent_run_visualization import visualize_agent_run


def execute():
    mod_path = os.path.dirname(os.path.abspath(__file__))

    # Read in environment configurations
    env_configs_file = PATH_PARAMS["configs"]
    env_configs = load_environment_configs(env_configs_file)

    # load the model
    path_to_model = os.path.join(mod_path, "logs", "best_model", "best_model.zip")
    model = RecurrentPPO.load(path_to_model)

    # get all scenarios in test folder
    path_to_scenarios = os.path.join(mod_path, "scenarios_validation")
    scenario_files = [join(path_to_scenarios, f) for f in listdir(path_to_scenarios) if isfile(join(path_to_scenarios, f))]

    # create the test environment
    test_env = AgentEnv(scenario_paths=scenario_files, env_configs=env_configs, test_env=True,
                        plot_agents=True, pick_random_scenario=False)

    # Iterate through each scenario
    for idx, scenario in enumerate(scenario_files):
        print(f"Start Scenario {scenario}")
        obs, _ = test_env.reset()
        done = False
        while not done:
            action, states = model.predict(obs)
            obs, rewards, done, truncated, info = test_env.step(action)
            if done:
                visualize_agent_run(test_env.simulation.log_path, mod_path)
                print(f"Scenario {scenario} completed")


if __name__ == '__main__':
    execute()
