__author__ = "Rainer Trauth, Alexander Hobmeier"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Release"

import os
import pathlib
import signal
import traceback
import pandas as pd

from cr_scenario_handler.simulation.simulation import Simulation
from cr_scenario_handler.utils.configuration_builder import ConfigurationBuilder
from stable_baselines3 import PPO

from frenetix_rl.evaluation.agent_run_visualization import visualize_agent_run
from frenetix_rl.evaluation.eval_plotting import box_plot, box_plot_harm, box_plot_risk
from frenetix_rl.evaluation.qualitative_plotting import plot_scenario, plot_agent_action
from frenetix_rl.gym_environment.environment.agent_env import AgentEnv
from frenetix_rl.gym_environment.paths import PATH_PARAMS
from frenetix_rl.utils.helper_functions import load_environment_configs
from frenetix_rl.evaluation.eval_helpers import Evaluator, EvalType
from frenetix_rl.evaluation.plot_scenario import plot_trajectory_difference


mod_path = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))


def timeout_handler(num, stack):
    print("Timeout")
    raise TimeoutError("Execution timed out")


def run_and_log_scenario_standard(scenario_path, timeout_scenario_mins=15):
    print(scenario_path)
    try:
        # set timeout for scenarios
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_scenario_mins*60)

        scenario_name = scenario_path.split("/")[-1].split(".")[0]
        log_path = mod_path + "/evaluation/logs_standard/" + str(scenario_name)
        scenario_folder = str(pathlib.Path(scenario_path).parent)

        config_sim = ConfigurationBuilder.build_sim_configuration(scenario_name, scenario_folder, mod_path)
        config_sim.simulation.use_multiagent = False

        config_planner = ConfigurationBuilder.build_frenetplanner_configuration(scenario_name, mod_path)
        config_planner.debug.use_cpp = True
        config_planner.debug.save_all_traj = True
        config_planner.debug.activate_logging = True
        config_planner.debug.multiproc = True
        config_planner.debug.log_risk = True

        simulation = Simulation(config_sim, config_planner)
        simulation.run_simulation()

    except:
        traceback.print_exc()

    return mod_path + "/logs/" + scenario_name


def run_and_log_scenario_rlplanner(scenario_path, model_path, timeout_scenario_mins=15, evaluate=False):
    try:
        # set timeout for scenarios
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_scenario_mins * 60)

        env_configs_file = PATH_PARAMS["configs"]
        # env_configs_file = os.path.join(mod_path, "", "gym_environment", "configs.yaml")
        env_configs = load_environment_configs(env_configs_file)

        # create the test environment
        test_env = AgentEnv(scenario_dir=None, env_configs=env_configs, test_env=True, evaluate_specific_scenario=scenario_path)

        # set log_path
        test_env.output_path = mod_path + "/evaluation/logs"

        if evaluate:
            test_env.config_planner.debug.activate_logging = True
            test_env.config_sim.visualization.save_plots = True

        # load the model
        model = PPO.load(model_path)

        # simulate the model
        obs, _ = test_env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, truncated, info = test_env.step(action)

        return test_env.log_path
    except:
        traceback.print_exc()


if __name__ == '__main__':
    path_scenarios = "/media/alex/Windows-SSD/Uni/9. Semester/failed/test"

    model_path = mod_path + "/logs/best_model/best_model"
    model_path = "/media/alex/Windows-SSD/Uni/9. Semester/2024-01-02/run_187/best_model/best_model.zip"

    path_standard_planner = "/media/alex/Windows-SSD/Uni/9. Semester/StandardPlanner/"

    evaluation = Evaluator("RLPlanner1", EvalType.RLPlanner)
    # Enable if one wants to create baseline with Standard planner
    #evaluation = Evaluator("StandardPlanner", EvalType.StandardPlanner)
    create_logs = False

    # ***************************************************
    # Create logs with RL Model
    # ***************************************************
    if create_logs:
        scenario_paths = []
        for r, d, f in os.walk(path_scenarios):
            for file in f:
                scenario_paths.append(os.path.join(r, file))

        for scenario_path in scenario_paths:
            scenario_name = scenario_path.split("/")[-1].split(".")[0]
            if evaluation.type == EvalType.RLPlanner:
                log_path = run_and_log_scenario_rlplanner(scenario_path, model_path)

            elif evaluation.type == EvalType.StandardPlanner:
                log_path = run_and_log_scenario_standard(scenario_path)
            else:
                print("Unknown Planner type")
            try:
                evaluation.evaluate(log_path, scenario_name)
            except:
                traceback.print_exc()

    # ***************************************************
    # Quantitative Plots
    # ***************************************************
    quantitative_path = mod_path + "/evaluation/plots/quantitative_plots/"
    if not os.path.exists(quantitative_path):
        os.makedirs(quantitative_path)
    paths_to_plot = {"Hybrid Planner":  mod_path + "/evaluation/RLPlanner1_evaluation.csv",
                     "Default Planner": path_standard_planner + "StandardPlanner_evaluation.csv"}
    columns_to_boxplot = ["mean_ego_risk", "mean_top_ego_risk", "mean_obst_risk", "mean_top_obst_risk", "max_ego_risk", "max_obst_risk", "mean_diff_to_ref_path", "std_obst_risk", "std_ego_risk", "mean_percentage_feasible_traj"]

    dfs = {}
    for key, path in paths_to_plot.items():
        dfs[key] = pd.read_csv(path, sep=';', header=0)

    for column in columns_to_boxplot:
        box_plot(dfs, column)
    box_plot_harm(dfs)

    columns_to_risk_plot = ["mean_ego_risk", "mean_top_ego_risk", "max_ego_risk", "std_ego_risk"]
    for column in columns_to_risk_plot:
        box_plot_risk(dfs, column, save_tikz=True)

    # ***************************************************
    # Qualitative Plots
    # ***************************************************
    for scenario_name in ["ZAM_Tjunction-1_14_T-1", "ZAM_Tjunction-1_86_T-1"]:
        scenario_path = path_scenarios + "/" + scenario_name + ".xml"
        #log_path = run_and_log_scenario_rlplanner(scenario_path, model_path, evaluate=True)

        paths_to_plot = {"Hybrid Planner": mod_path + "/evaluation/logs/" + scenario_name,
            "Default Planner": path_standard_planner + scenario_name}

        dfs = {}
        for key, path in paths_to_plot.items():
            dfs[key] = pd.read_csv(path + "/logs.csv", sep=';', header=0)

        columns = ["ego_risk", "obst_risk", "percentage_feasible_traj", "velocities_mps", "d_position_m",
                   "s_position_m"]
        for column in columns:
            plot_scenario(dfs, column, scenario_name)

        # plot action progression
        df_action = pd.read_csv(paths_to_plot["RL Planner"] + "/agent_logs.csv", sep=';', header=0)
        for action in ["prediction_action"]:
            plot_agent_action(df_action, action, scenario_name)

        visualize_agent_run(paths_to_plot["RL Planner"], mod_path)

        plot_trajectory_difference(scenario_path, scenario_name, dfs, timesteps_to_plot=[60, 70, 80, 90, 100])

