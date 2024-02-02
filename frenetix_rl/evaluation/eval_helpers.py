__author__ = "Rainer Trauth, Alexander Hobmeier"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Release"

import json
import os
from enum import Enum

import numpy as np
import pandas as pd
from commonroad.scenario.obstacle import ObstacleType
from risk_assessment.helpers.collision_helper_function import angle_range
from risk_assessment.helpers.harm_parameters import HarmParameters
from risk_assessment.helpers.properties import get_obstacle_mass
from risk_assessment.utils.logistic_regression import get_protected_log_reg_harm, get_unprotected_log_reg_harm
from risk_assessment.utils.logistic_regression_symmetrical import get_protected_inj_prob_log_reg_ignore_angle

mod_path = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))


class EvalType(Enum):
    StandardPlanner = 1
    RLPlanner = 2


class Evaluator:
    def __init__(self, name_planner: str, type: EvalType):
        self.path_logs = "../evaluation"
        self.type = type

        self.header = (
            "scenario_name;"
            "success;"
            "ego_harm;"
            "obst_harm;"
            "mean_ego_risk;"
            "mean_top_ego_risk;"
            "std_ego_risk;"
            "max_ego_risk;"
            "min_ego_risk;"
            "total_ego_risk;"
            "mean_obst_risk;"
            "mean_top_obst_risk;"
            "std_obst_risk;"
            "max_obst_risk;"
            "min_obst_risk;"
            "total_obst_risk;"
            "mean_percentage_feasible_traj;"
            "std_percentage_feasible_traj;"
            "mean_diff_to_ref_path;"
            "std_diff_to_ref_path;"
            "max_diff_to_ref_path;"

        )

        evaluation_file_name = name_planner + "_evaluation.csv"
        self.__log_path = os.path.join(mod_path, self.path_logs, evaluation_file_name)

        # write header to logging file
        if not os.path.exists(self.__log_path):
            with open(self.__log_path, "w+") as fh:
                fh.write(self.header)

        self.obstacle_protection = {
            ObstacleType.CAR: True,
            ObstacleType.TRUCK: True,
            ObstacleType.BUS: True,
            ObstacleType.BICYCLE: False,
            ObstacleType.PEDESTRIAN: False,
            ObstacleType.PRIORITY_VEHICLE: True,
            ObstacleType.PARKED_VEHICLE: True,
            ObstacleType.TRAIN: True,
            ObstacleType.MOTORCYCLE: False,
            ObstacleType.TAXI: True,
            ObstacleType.ROAD_BOUNDARY: None,
            ObstacleType.PILLAR: None,
            ObstacleType.CONSTRUCTION_ZONE: None,
            ObstacleType.BUILDING: None,
            ObstacleType.MEDIAN_STRIP: None,
            ObstacleType.UNKNOWN: False,
        }

    def calculate_harm(self, df_collision, df_ego):
        # get data from logs
        ego_pos_last = [df_ego["x_position_vehicle_m"].iloc[-2], df_ego["y_position_vehicle_m"].iloc[-2]]
        ego_pos = [df_ego["x_position_vehicle_m"].iloc[-1], df_ego["y_position_vehicle_m"].iloc[-1]]
        ego_velocity = float(df_ego["velocities_mps"].iloc[-1].split(",")[0])
        delta_ego_pos = np.array(ego_pos) - np.array(ego_pos_last)
        ego_yaw = np.arctan2(delta_ego_pos[1], delta_ego_pos[0])

        # get parameters
        with open(os.path.join("../configurations/risk.json"), "r") as f:
            modes = json.load(f)
        with open(os.path.join("../configurations/harm_parameters.json"), "r") as f:
            coeffs = json.load(f)

        if df_collision["center_x"].iloc[0] == "None":
            ego_harm = get_protected_inj_prob_log_reg_ignore_angle(
                velocity=ego_velocity, coeff=coeffs
            )
            return ego_harm, 0

        else:
            obs_pos = [df_collision["center_x"].iloc[0], df_collision["center_y"].iloc[0]]
            obs_pos_last = [df_collision["last_center_x"].iloc[0], df_collision["last_center_y"].iloc[0]]
            pos_delta = np.array(obs_pos) - np.array(obs_pos_last)
            obstacle_velocity = np.linalg.norm(pos_delta) / 0.1
            obstacle_yaw = np.arctan2(pos_delta[1], pos_delta[0])
            obstacle_size = 4 * df_collision["r_x"].iloc[0] * df_collision["r_y"].iloc[0]

            # calculate crash angle
            pdof = angle_range(obstacle_yaw - ego_yaw + np.pi)
            rel_angle = np.arctan2(
                obs_pos_last[1] - ego_pos_last[1], obs_pos_last[0] - ego_pos_last[0]
            )
            ego_angle = angle_range(rel_angle - ego_yaw)
            obs_angle = angle_range(np.pi + rel_angle - obstacle_yaw)
            # create dictionaries with crash relevant parameters
            ego_vehicle = HarmParameters()
            obstacle = HarmParameters()

            # assign parameters to dictionary
            # TODO make ego and obstacle type dynamic
            ego_vehicle.type = ObstacleType.CAR
            obstacle.type = ObstacleType.CAR
            ego_vehicle.protection = self.obstacle_protection[ego_vehicle.type]
            obstacle.protection = self.obstacle_protection[obstacle.type]
            if ego_vehicle.protection is not None:
                ego_vehicle.velocity = ego_velocity
                ego_vehicle.yaw = ego_yaw
                ego_vehicle.size = df_collision["ego_length"].iloc[0] * df_collision["ego_width"].iloc[0]
                ego_vehicle.mass = get_obstacle_mass(
                    obstacle_type=ego_vehicle.type, size=ego_vehicle.size
                )
            else:
                ego_vehicle.mass = None
                ego_vehicle.velocity = None
                ego_vehicle.yaw = None
                ego_vehicle.size = None

            if obstacle.protection is not None:
                obstacle.velocity = obstacle_velocity
                obstacle.yaw = obstacle_yaw
                obstacle.size = obstacle_size
                obstacle.mass = get_obstacle_mass(
                    obstacle_type=obstacle.type, size=obstacle.size
                )
            else:
                obstacle.mass = None
                obstacle.velocity = None
                obstacle.yaw = None
                obstacle.size = None

            if obstacle.protection is True:
                ego_vehicle.harm, obstacle.harm = get_protected_log_reg_harm(
                    ego_vehicle=ego_vehicle,
                    obstacle=obstacle,
                    pdof=pdof,
                    ego_angle=ego_angle,
                    obs_angle=obs_angle,
                    modes=modes,
                    coeffs=coeffs,
                )
            elif obstacle.protection is False:
                ego_vehicle.harm, obstacle.harm = get_unprotected_log_reg_harm(
                    ego_vehicle=ego_vehicle, obstacle=obstacle, pdof=pdof, coeff=coeffs
                )
            else:
                ego_vehicle.harm = 1
                obstacle.harm = 1

        return ego_vehicle.harm, obstacle.harm

    def evaluate(self, path, scenario_name):
        df_ego = pd.read_csv(path+"/logs.csv", sep=';', header=0)
        new_line = "\n" + str(scenario_name)
        # log scenario success and collision parameters
        try:
            df_collision = pd.read_csv(path+"/collision.csv", sep=';', header=0)
            new_line += ";" + str(False)

            # get collision harm
            ego_harm, obstacle_harm = self.calculate_harm(df_collision, df_ego)
            new_line += ";" + str(ego_harm)
            new_line += ";" + str(obstacle_harm)
        except FileNotFoundError:
            new_line += ";" + str(True)
            new_line += ";" + str(0)
            new_line += ";" + str(0)

        # log risk
        ego_risk = np.sort(np.array(df_ego["ego_risk"]))
        obst_risk = np.sort(np.array(df_ego["obst_risk"]))
        new_line += ";" + str(np.mean(ego_risk))
        new_line += ";" + str(np.mean(ego_risk[-10:]))
        new_line += ";" + str(np.std(ego_risk))
        new_line += ";" + str(np.max(ego_risk))
        new_line += ";" + str(np.min(ego_risk))
        new_line += ";" + str(np.sum(ego_risk))

        new_line += ";" + str(np.mean(obst_risk))
        new_line += ";" + str(np.mean(obst_risk[-10:]))
        new_line += ";" + str(np.std(obst_risk))
        new_line += ";" + str(np.max(obst_risk))
        new_line += ";" + str(np.min(obst_risk))
        new_line += ";" + str(np.sum(obst_risk))

        # log percentage feasible trajectories
        d_value = np.array(df_ego["percentage_feasible_traj"])
        new_line += ";" + str(np.mean(d_value))
        new_line += ";" + str(np.std(d_value))

        # log distance to ref path
        d_value = np.array(df_ego["d_position_m"])
        new_line += ";" + str(np.mean(d_value))
        new_line += ";" + str(np.std(d_value))
        new_line += ";" + str(np.max(d_value))

        with open(self.__log_path, "a") as fh:
            fh.write(new_line)
