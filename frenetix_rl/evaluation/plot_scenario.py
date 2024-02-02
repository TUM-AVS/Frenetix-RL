__author__ = "Rainer Trauth, Alexander Hobmeier"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Release"

import os

import numpy as np
import pandas as pd
from commonroad.common.file_reader import CommonRoadFileReader

from commonroad.visualization.draw_params import ShapeParams
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.geometry.shape import Rectangle
from frenetix_rl.evaluation.qualitative_plotting import tikzplotlib_fix_ncols
from matplotlib import pyplot as plt

mod_path = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))


def plot_trajectory_difference(scenario_path, scenario_name, dfs, timesteps_to_plot=[], plot_window=15, save=True, save_tikz=False, show=False):
    # reset plotting
    plt.clf()
    # open scenario
    if timesteps_to_plot is None:
        timesteps_to_plot = []
    scenario, planning_problem_set = CommonRoadFileReader(scenario_path).open()
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    # create renderer object (if no existing renderer is passed)
    rnd = MPRenderer(figsize=(20, 10))

    # set plotting window
    if plot_window is not None:
        # focus on window around all agents
        x_coord = 11
        y_coord = 0.5
        rnd.plot_limits = [-plot_window + x_coord,
                           plot_window + x_coord,
                           -plot_window + y_coord,
                           plot_window + y_coord]

    # set renderer draw params
    rnd.draw_params.time_begin = 0# if len(timesteps_to_plot) == 0 else timesteps_to_plot[-1]
    rnd.draw_params.planning_problem.initial_state.state.draw_arrow = False
    rnd.draw_params.planning_problem.initial_state.state.radius = 0.5
    rnd.draw_params.axis_visible = False
    rnd.draw_params.dynamic_obstacle.trajectory.draw_trajectory = False

    # set occupancy shape params
    occ_params = ShapeParams()
    occ_params.facecolor = '#E37222'
    occ_params.edgecolor = '#9C4100'
    occ_params.opacity = 1.0
    occ_params.zorder = 51

    # visualize scenario
    scenario.draw(rnd)
    # visualize planning problem
    planning_problem.draw(rnd)

    # visualize obstacle
    for i in range(147):
        # state = agent.record_state_list[i]
        center = scenario.dynamic_obstacles[0].prediction.trajectory.state_list[i].position
        orientation = scenario.dynamic_obstacles[0].prediction.trajectory.state_list[i].orientation
        occ_pos = Rectangle(length=4.6, width=1.8,
                            center=np.array(center),
                            orientation=orientation)

        # get color for heatmap
        occ_params.facecolor = "#005293"
        occ_params.edgecolor = "#000000"
        if i in timesteps_to_plot:
            occ_params.facecolor = "#98C6EA"
        occ_pos.draw(rnd, draw_params=occ_params)

    # render scenario and occupancies
    rnd.render()

    keys = []
    for key in dfs.keys():
        keys.append(key)
    data1 = dfs[keys[0]]
    data2 = dfs[keys[1]]

    # visualize trajectory
    rnd.ax.plot(data1["x_position_vehicle_m"], data1["y_position_vehicle_m"], color="#003359", marker='.', markersize=10, zorder=21, linewidth=2.5, label=keys[0])
    rnd.ax.plot(data2["x_position_vehicle_m"], data2["y_position_vehicle_m"], color="#E37222", marker='.', markersize=10, zorder=21, linewidth=2.5, label=keys[1])
    rnd.ax.legend()

    for timestep in timesteps_to_plot:
        # plot timestep data
        if timestep < len(data1):
            rnd.ax.plot([data1.iloc[timestep]["x_position_vehicle_m"]], [data1.iloc[timestep]["y_position_vehicle_m"]], color="#003359", marker='|', markersize=30, zorder=21, linewidth=5)
            rnd.ax.text(data1.iloc[timestep]["x_position_vehicle_m"]-0.5, data1.iloc[timestep]["y_position_vehicle_m"]+0.8, str(timestep), color="black", fontsize=15, zorder=21)

        if timestep < len(data2):
            rnd.ax.plot([data2.iloc[timestep]["x_position_vehicle_m"]], [data2.iloc[timestep]["y_position_vehicle_m"]], color="#E37222", marker='|', markersize=30, zorder=21, linewidth=5)
            rnd.ax.text(data2.iloc[timestep]["x_position_vehicle_m"]-0.5, data2.iloc[timestep]["y_position_vehicle_m"]-1.5, str(timestep), color="black", fontsize=15, zorder=21)

        if timestep > 30:
            center = scenario.dynamic_obstacles[0].prediction.trajectory.state_list[timestep].position
            rnd.ax.text(center[0]+2.5, center[1]+0.85, str(timestep), color="black", fontsize=15, zorder=21)

    save_path = mod_path + "/evaluation/plots/qualitative_plots/" + scenario_name
    if save:
        plt.savefig(save_path + "/planner_trajectory.pdf", bbox_inches='tight')
    if save_tikz:
        import tikzplotlib
        tikzplotlib_fix_ncols(plt.gcf())
        tikzplotlib.save(save_path + "/planner_trajectory.tikz")
    if show:
        plt.show()

def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    # Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]


def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]


def plot_trajectory_cost(scenario_path, scenario_name, log_path, action_log_path, plot_window=40):
    plt.clf()
    # open scenario
    scenario, planning_problem_set = CommonRoadFileReader(scenario_path).open()
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    # create renderer object (if no existing renderer is passed)
    rnd = MPRenderer(figsize=(20, 10))

    if plot_window is not None:
        # focus on window around all agents
        x_coord = 11
        y_coord = 0.5
        rnd.plot_limits = [-plot_window + x_coord,
                           plot_window + x_coord,
                           -plot_window + y_coord,
                           plot_window + y_coord]

    # set renderer draw params
    rnd.draw_params.time_begin = -1
    rnd.draw_params.planning_problem.initial_state.state.draw_arrow = False
    rnd.draw_params.planning_problem.initial_state.state.radius = 0.5
    rnd.draw_params.axis_visible = False
    rnd.draw_params.dynamic_obstacle.trajectory.draw_trajectory = False

    # set occupancy shape params
    occ_params = ShapeParams()
    occ_params.opacity = 0.8
    occ_params.zorder = 51

    # visualize scenario
    scenario.draw(rnd)

    # get data
    data = pd.read_csv(log_path, sep=';', header=0)
    action_data = pd.read_csv(action_log_path, sep=';', header=0)

    # get colors
    colors = get_color_gradient("#005293", "#E37222",  100)

    color_data = np.cumsum(action_data["prediction_action"])
    normalizedData = (color_data-np.min(color_data))/(np.max(color_data)-np.min(color_data))*99
    normalizedData = normalizedData.astype(int)

    # visualize occupancies of trajectory
    for i in range(0, min(len(data), len(normalizedData)), 2):
        # state = agent.record_state_list[i]
        occ_pos = Rectangle(length=4.6, width=1.8,
                            center=np.array([data.iloc[i]["x_position_vehicle_m"], data.iloc[i]["y_position_vehicle_m"]]),
                            orientation=float(data.iloc[i]["theta_orientations_rad"].split(",")[0]))

        # get color for heatmap
        print(normalizedData[i])
        color = colors[normalizedData[i]]
        occ_params.facecolor = color
        occ_params.edgecolor = color
        occ_pos.draw(rnd, draw_params=occ_params)

    # render scenario and occupancies
    rnd.render()

    plt.savefig("/media/alex/Windows-SSD/Uni/9. Semester/frenetix-rl/evaluation/" + str(scenario_name) + "_trajectory_cost.svg", format='svg',
                    bbox_inches='tight')


def plot_trajectory_top_risk(scenario_path, scenario_name, log_path, action_log_path, plot_window=20, n=10):
    plt.clf()
    # open scenario
    scenario, planning_problem_set = CommonRoadFileReader(scenario_path).open()
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    # create renderer object (if no existing renderer is passed)
    rnd = MPRenderer(figsize=(20, 10))

    if plot_window is not None:
        # focus on window around all agents
        x_coord = 11
        y_coord = 0.5
        rnd.plot_limits = [-plot_window + x_coord,
                           plot_window + x_coord,
                           -plot_window + y_coord,
                           plot_window + y_coord]

    # set renderer draw params
    rnd.draw_params.time_begin = -1
    rnd.draw_params.planning_problem.initial_state.state.draw_arrow = False
    rnd.draw_params.planning_problem.initial_state.state.radius = 0.5
    rnd.draw_params.axis_visible = False
    rnd.draw_params.dynamic_obstacle.trajectory.draw_trajectory = False

    # set occupancy shape params
    occ_params = ShapeParams()
    occ_params.opacity = 0.8
    occ_params.zorder = 51

    # visualize scenario
    scenario.draw(rnd)

    # get data
    data = pd.read_csv(log_path, sep=';', header=0)
    action_data = pd.read_csv(action_log_path, sep=';', header=0)

    # get colors
    colors = get_color_gradient("#005293", "#E37222",  100)

    color_data = np.cumsum(action_data["prediction_action"])
    normalizedData = (color_data-np.min(color_data))/(np.max(color_data)-np.min(color_data))*99
    normalizedData = normalizedData.astype(int)

    color_data = np.argpartition(data["ego_risk"], -n)[-n:]

    # visualize obstacle
    for i in range(147):
        # state = agent.record_state_list[i]
        center = scenario.dynamic_obstacles[0].prediction.trajectory.state_list[i].position
        orientation = scenario.dynamic_obstacles[0].prediction.trajectory.state_list[i].orientation
        occ_pos = Rectangle(length=4.6, width=1.8,
                            center=np.array(center),
                            orientation=orientation)

        # get color for heatmap
        occ_params.facecolor = "#005293"
        if i in color_data:
            occ_params.facecolor = "#E37222"
        occ_params.edgecolor = "#000000"
        occ_pos.draw(rnd, draw_params=occ_params)


    # visualize occupancies of trajectory
    for i in range(0, min(len(data), len(normalizedData))):
        # state = agent.record_state_list[i]
        occ_pos = Rectangle(length=4.6, width=1.8,
                            center=np.array([data.iloc[i]["x_position_vehicle_m"], data.iloc[i]["y_position_vehicle_m"]]),
                            orientation=float(data.iloc[i]["theta_orientations_rad"].split(",")[0]))

        # get color for heatmap
        #print(normalizedData[i])
        #color = colors[normalizedData[i]]
        print(i, i in color_data)
        if i in color_data:
            color = "#E37222"
        else:
            color = "#003359"
        occ_params.facecolor = color
        occ_params.edgecolor = "#000000"
        occ_pos.draw(rnd, draw_params=occ_params)

    # render scenario and occupancies
    rnd.render()

    plt.savefig("/media/alex/Windows-SSD/Uni/9. Semester/frenetix-rl/evaluation/" + str(scenario_name) + "_top_risk.svg", format='svg',
                    bbox_inches='tight')

