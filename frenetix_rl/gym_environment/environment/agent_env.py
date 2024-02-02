__author__ = "Rainer Trauth, Alexander Hobmeier"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Release"

# standard imports
import os
import random
import signal
import sys
import traceback
import warnings
from typing import Any
import logging
import datetime
import time

import gymnasium as gym
import numpy as np
from shapely.set_operations import intersection

from commonroad.common.file_reader import CommonRoadFileReader

# scenario handler
from cr_scenario_handler.utils.configuration_builder import ConfigurationBuilder
from cr_scenario_handler.simulation.simulation import Simulation

# Reinforcement Learner
from frenetix_rl.gym_environment.reward.termination import Termination
from frenetix_rl.gym_environment.reward import reward_constructor
from frenetix_rl.gym_environment.observation import ObservationCollector
import frenetix_rl.gym_environment.utils.simulation_helpers as sh
import frenetix_rl.utils.logging_helpers as lh

__all__ = ["AgentEnv"]
LOGGER = logging.getLogger(__name__)


class AgentEnv(gym.Env):
    def __init__(self, scenario_paths,
                 env_configs,
                 test_env=False,
                 pick_random_scenario=True,
                 remove_scenarios=False,
                 plot_agents=False):
        """Represents a single-agent simulation of a commonroad scenario.

                Manages the agent's local view on the scenario, the planning problem,
                planner interface, collision detection, and per-agent plotting and logging.
                Contains the step function of the agent.

                :param scenario_paths: List(str), list of paths to scenarios that will be executed in this scenario
                :param env_configs, configs for the environment
                :param test_env: boolean, if this is a test environment
                """

        # Environment Parameters
        self.general_configs = env_configs
        self.test_env = test_env

        # set action and observation space to superclass
        self.observation_collector = ObservationCollector(self.general_configs)
        self.observation_space = self.observation_collector.observation_space
        self.action_space = sh.construct_action_space(1 + len(self.general_configs["action_configs"]["cost_terms"]))

        self.current_timestep = None

        # ################################
        # Get all available scenario paths
        # ################################
        self.scenario_paths = scenario_paths

        self.scenario_index = -1
        self.pick_random_scenario = pick_random_scenario
        self.remove_scenarios = remove_scenarios
        self.scenario_iteration = 0
        self.scenarios_executed = 0
        self.status = True
        self.cost_weights = None
        self.agent_logger = None

        # ################################
        # Get all Planner Configurations
        # ################################
        self.mod_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))))

        scenario_folder = os.path.join(self.mod_path, "scenarios")
        self.config_sim = ConfigurationBuilder.build_sim_configuration("default", scenario_folder,
                                                                       self.mod_path)
        self.config_planner = ConfigurationBuilder.build_frenetplanner_configuration("default",
                                                                                     root_path=self.mod_path)
        self.config_planner.debug.use_cpp = True

        self.output_path = os.path.join(self.config_sim.simulation.mod_path, self.config_sim.simulation.path_output)
        self.log_path = None
        self.simulation = None
        self.agent = None

        if plot_agents:
            self.config_sim.visualization.save_all_individual_plots = False

        signal.signal(signal.SIGALRM, self.signal_handler)

        # ################################
        # Initialize Action Parameter
        # ################################
        action_configs = self.general_configs["action_configs"]
        self.cost_terms = action_configs["cost_terms"]
        self.weight_low = action_configs["weight_low"]
        self.weight_high = action_configs["weight_high"]
        self.weight_update_low = action_configs["weight_update_low"]
        self.weight_update_high = action_configs["weight_update_high"]
        self.prediction_weight_low = action_configs["weight_prediction_low"]
        self.prediction_weight_high = action_configs["weight_prediction_high"]
        self.prediction_weight_update_low = action_configs["weight_prediction_update_low"]
        self.prediction_weight_update_high = action_configs["weight_prediction_update_high"]
        # self.sample_d_low = action_configs["sample_d_low"]
        # self.sample_d_high = action_configs["sample_d_high"]
        # self.sample_t_low = action_configs["sample_t_low"]
        # assert self.sample_t_low >= 0.1, "t_min cant be <= 0"
        # self.sample_t_high = self.config_planner.planning.planning_horizon

        self._set_rescale_weight_factor_and_bias()
        self._set_rescale_prediction_weight_factor_and_bias()
        # self._set_rescale_sample_d_factor_and_bias()
        # self._set_rescale_sample_t_factor_and_bias()

        # initialize gym parameter collector
        self.termination = Termination(self.general_configs)
        self.reward_function = reward_constructor.make_reward(self.general_configs)
        self.termination_reason = None

    def signal_handler(self, signum, frame):
        raise TimeoutError("It took too long to load the scenario, loading new scenario")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        observation = None
        while observation is None:
            # try to load new scenario
            try:
                # set timeout, load new scenario if scenario loading takes too long
                signal.alarm(self.general_configs["load_scenario_timeout"])

                # #####################################################
                # Initialize Environment including Scenario and Planner
                # #####################################################
                self.simulation_preprocessing()

                # ################################
                # Prepare Gym Parameter Collectors
                # ################################
                self.observation_collector.reset(self.agent)
                self.reward_function.reset(self.test_env, self.agent.planner_interface.planner.cost_weights)
                self.termination.reset()

                # reset current timestep
                self.current_timestep = -1
                self.status = True

                observation = self.observe()

                # end timer for scenario loading
                signal.alarm(0)
            except:
                traceback.print_exc()

        # count in which eval iteration the environment is
        self.scenarios_executed += 1
        if self.test_env and self.scenarios_executed % (
                self.general_configs["training_configs"]["n_eval_episodes"] + 1) == 0:
            self.scenarios_executed = 0
            self.scenario_iteration += 1

        # create action logging files
        if self.test_env:
            self.agent_logger = lh.AgentLogger(self.log_path, self.general_configs["action_configs"]["cost_terms"])

        return observation, dict()

    def simulation_preprocessing(self):
        """ Modify a commonroad scenario to prepare it for the simulation.

        Reads the scenario and planning problem from the configuration,
        selects the agents to be simulated, and creates missing planning problems and
        dummy obstacles.
        """
        if self.pick_random_scenario:
            self.scenario_index = random.randint(0, len(self.scenario_paths)-1)
        else:
            self.scenario_index += 1
            if self.scenario_index >= len(self.scenario_paths):
                self.scenario_index = 0
        scenario_path = self.scenario_paths[self.scenario_index]
        if self.remove_scenarios:
            self.scenario_paths.pop(self.scenario_index)

        # Only dummy function to read out agent ID of ego vehicle
        _, planning_problem_set = CommonRoadFileReader(scenario_path).open()
        agent_id = list(planning_problem_set.planning_problem_dict.values())[0].planning_problem_id

        # ######################################################
        # Set additional configs to setup simulation environment
        # ######################################################
        self.config_sim.simulation.name_scenario = scenario_path.split("/")[-1].replace(".xml", "")
        self.config_sim.simulation.path_output = os.path.join(self.output_path, self.config_sim.simulation.name_scenario)
        self.config_sim.simulation.scenario_path = scenario_path
        if self.test_env:
            self.log_path = os.path.join(self.output_path, str(self.config_sim.simulation.name_scenario))
            self.config_planner.debug.activate_logging = True
            self.config_planner.debug.multiproc = False  # For parallel execution
        else:
            self.log_path = os.path.join(self.output_path, "train_scenario", f"{agent_id}")
        self.config_sim.simulation.log_path = self.log_path
        self.config_sim.simulation.use_multiagent = False

        print("Scenario Name: " + str(self.config_sim.simulation.name_scenario))

        self.simulation = Simulation(self.config_sim, self.config_planner)
        self.agent = self.simulation.agents[0]

        self.cost_weights = self.agent.planner_interface.planner.cost_weights
        self.cost_weights = {key: 0.0 for key in self.cost_weights}

        if not len(self.simulation.agents) == 1:
            raise EnvironmentError("Run should only execute one agent per simulation!")

    def set_action(self, action):
        """
        function that takes the action from the Environment and adjusts the weights of the planner accordingly
        Args:
            action: vector of weights from the environment
        Returns:

        """
        # decode action
        action = self.rescale_action(action)

        prediction_weight = np.float64(action[0])
        weights = action[1:]

        # set cost weights
        self.cost_weights = {key: np.float64(0.0) for key in self.cost_weights}
        self.cost_weights["prediction"] = prediction_weight
        for i, name in enumerate(self.cost_terms):
            if name in self.cost_terms:
                self.cost_weights[name] = np.float64(weights[self.cost_terms.index(name)])
        self.agent.planner_interface.planner.handler.set_all_cost_weights_to_zero()
        self.agent.planner_interface.planner.cost_weights = self.cost_weights
        self.agent.planner_interface.planner.trajectory_handler_set_constant_cost_functions()

    def set_action_cummulative(self, action):
        """
        function that takes the action from the Environment and adjusts the weights of the planner accordingly
        Args:
            action: vector of weights from the environment
        Returns:

        """
        # decode action
        action = self.rescale_action(action)

        prediction_weight = action[0]
        weights = action[1:]

        previous_weights = self.agent.planner_interface.planner.cost_weights

        # set cost weights
        self.cost_weights = {key: np.float64(0.0) for key in self.cost_weights}
        self.cost_weights["prediction"] = np.clip(previous_weights["prediction"] + prediction_weight,
                                                  a_min=self.prediction_weight_low, a_max=self.prediction_weight_high)
        for i, name in enumerate(self.cost_terms):
            if name in self.cost_terms:
                self.cost_weights[name] = np.clip(np.float64(previous_weights[name] + weights[self.cost_terms.index(name)]),
                                                  a_min=self.weight_low, a_max=self.weight_high)
        self.agent.planner_interface.planner.handler.set_all_cost_weights_to_zero()
        self.agent.planner_interface.planner.cost_weights = self.cost_weights
        self.agent.planner_interface.planner.trajectory_handler_set_constant_cost_functions()

    def step(self, action):

        self.current_timestep += 1
        self.set_action_cummulative(action)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value encountered in intersection", RuntimeWarning,
                                    module=intersection.__module__)
            try:
                self.simulation.global_timestep += 1
                self.simulation.process_times = {}
                step_time_start = time.perf_counter()
                self.status = self.simulation.step_sequential_simulation()
                self.simulation.visualize_simulation(self.simulation.global_timestep)
                self.simulation.process_times.update({"total_sim_step": time.perf_counter() - step_time_start})

                # get batch process times
                self.simulation.process_times[self.simulation.batch_list[0].name] = self.simulation.batch_list[0].process_times

                observation, reward, terminated, info = self.get_gym_parameter(action)
            except Exception as e:
                # Capture the exception information
                exc_info = traceback.format_exc()

                # Print the traceback to stderr (optional, but often useful)
                print(exc_info, file=sys.stderr)

                # Define a filename for the exception log
                exception_filename = "exception_log.txt"

                # Open the file in append mode, so you don't overwrite previous exceptions
                with open(os.path.join(self.output_path, exception_filename), "a") as file:
                    # Write a header for the exception with a timestamp
                    file.write(f"Exception occurred on {datetime.datetime.now()}\n")
                    file.write(f"Scenario executed was: {str(self.agent.scenario.scenario_id)}\n")
                    # Write the traceback information
                    file.write(exc_info)
                    # Optionally add a separator for readability
                    file.write("-" * 80 + "\n\n")

                # Set default values in case of an exception
                observation = np.zeros(self.observation_collector.observation_space_size)
                terminated = True
                reward = self.general_configs["reward_configs"]["sparse_reward"]["reward_exception"]
                info = dict()

        # Logging agent infos for test_env
        if self.test_env:
            self.agent_logger.log_progress(self.simulation.global_timestep, action, info["reward_list"])

        return observation, float(reward), terminated, terminated, info

    def get_gym_parameter(self, action):
        """
        Function that calculates all parameters needed for the gym environment to train
        Args:
            action:

        Returns:
            observation:
            reward:
            terminated:
            info:
        """
        observation = self.observation_collector.observe(self)

        # **************************
        # Check Termination
        # **************************
        done, reason, termination_info = self.termination.is_terminated(self.agent,
                                                                        self.observation_collector.observation_dict)

        if reason is not None:
            self.termination_reason = reason

        if done:
            terminated = True
            termination_info_path = os.path.join(self.mod_path, "logs", "termination_info_log.txt")
            with open(termination_info_path, "a") as termination_file:
                termination_file.write(str(self.agent.scenario.scenario_id) + ": " + self.termination_reason + "\n")
        else:
            terminated = False

        # in case termination is false, but agent returns termination, set termination to true
        if not self.status:
            terminated = True

        # **************************
        # Calculate Reward
        # **************************
        rewards = self.reward_function.calc_reward(self.observation_collector.observation_dict, action, self.agent,
                                                   self.config_sim, self.simulation, termination_info)
        # **************************
        # Collect Info
        # **************************
        info = {
            "scenario_name": self.config_sim.simulation.name_scenario,
            "chosen_action": action,
            "current_episode_time_step": self.simulation.global_timestep,
            "max_episode_time_steps": self.observation_collector.goal_observation.episode_length,
            "termination_reason": self.termination_reason,
            "observation_dict": self.observation_collector.observation_dict,
            "reward_list": rewards
        }
        info.update(termination_info)

        return observation, np.sum(rewards), terminated, info

    def observe(self):
        return self.observation_collector.observe(self)

    def rescale_action(self, action: np.ndarray):
        """
        Rescales the normalized action from [-1,1] to the required range

        :param action: action from the Gym Environment.
        :return: rescaled action

        actions:
            action[0] -> Prediction
            action[1::] -> Costs

        """

        # prediction cost weight
        action[0] = action[0] * self._rescale_prediction_weight_factor + self._rescale_prediction_weight_bias
        assert self.general_configs["action_configs"]["weight_prediction_update_low"] <= action[0] <= \
               self.general_configs["action_configs"]["weight_prediction_update_high"], \
               "Action value for weight_prediction is out of bounds!"
        # cost weights
        action[1:] = action[1:] * self._rescale_weight_factor + self._rescale_weight_bias

        # d sampling
        # action[0] = action[0] * self._rescale_sample_d_factor + self._rescale_sample_d_bias
        # assert self.general_configs["action_configs"]["sample_d_low"] <= action[0] <= \
        #        self.general_configs["action_configs"]["sample_d_high"], \
        #        "Action value for d_sampling is out of bounds!"

        # t sampling
        # action[1] = action[1] * self._rescale_sample_t_factor + self._rescale_sample_t_bias
        return action

    def _set_rescale_weight_factor_and_bias(self):
        self._rescale_weight_factor: np.float64 = (self.weight_update_high - self.weight_update_low) / 2.
        self._rescale_weight_bias: np.float64 = self.weight_update_high - self._rescale_weight_factor

    def _set_rescale_prediction_weight_factor_and_bias(self):
        self._rescale_prediction_weight_factor: np.float64 = (self.prediction_weight_update_high - self.prediction_weight_update_low) / 2.
        self._rescale_prediction_weight_bias: np.float64 = self.prediction_weight_update_high - self._rescale_prediction_weight_factor

    # def _set_rescale_sample_d_factor_and_bias(self):
    #     self._rescale_sample_d_factor = (self.sample_d_high - self.sample_d_low) / 2.
    #     self._rescale_sample_d_bias = self.sample_d_high - self._rescale_sample_d_factor

    # def _set_rescale_sample_t_factor_and_bias(self):
    #    self._rescale_sample_t_factor = (self.sample_t_high - self.sample_t_low) / 2.
    #    self._rescale_sample_t_bias = self.sample_t_high - self._rescale_sample_t_factor
