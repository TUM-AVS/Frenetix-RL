__author__ = "Rainer Trauth, Alexander Hobmeier"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Release"

import os
import matplotlib.pyplot as plt
import numpy as np

mod_path = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def plot_scenario(dfs, column, scenario_name, show=False, save=True, save_tikz=False):
    plt.rcParams.update({'font.size': 15})
    data = []
    keys = []
    for key, df in dfs.items():
        column_data = np.array(df[column])
        if column_data.dtype == object:
            for i in range(len(column_data)):
                column_data[i] = float(column_data[i].split(",")[0])
            column_data = column_data.astype(float)
        data.append(column_data)
        keys.append(key)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))

    # plot actual values
    axes[0].plot(data[0], color="#003359")
    axes[0].plot(data[1], color="#98C6EA")
    axes[0].legend(keys)
    axes[0].set(xlabel="Timesteps (in 0.1s)")
    axes[0].set(ylabel=column.replace("_", " ").title())

    # plot differences
    min_len = min(len(data[0]), len(data[1]))
    max_len = max(len(data[0]), len(data[1]))
    axes[1].plot(data[0][:min_len]-data[1][:min_len], color="#005293")
    axes[1].set(xlabel="Timesteps (in 0.1s)")
    axes[1].set(ylabel="Difference")

    # save figures
    save_path = mod_path + "/evaluation/plots/qualitative_plots/" + scenario_name
    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + "/" + column + "_qualitative_analysis.pdf")
    # only works in python<=3.10
    if save_tikz:
        import tikzplotlib
        tikzplotlib_fix_ncols(fig)
        tikzplotlib.save(save_path + "/" + column + "_qualitative_analysis.tikz")
    if show:
        plt.show()


def plot_agent_action(df_agent_log, column, scenario_name, show=False, save=True, save_tikz=False):
    plt.clf()
    plt.rcParams.update({'font.size': 15})

    # plot actual values
    plt.plot(np.cumsum(df_agent_log[column]), color="#003359")

    # save figures
    save_path = mod_path + "/evaluation/plots/qualitative_plots/" + scenario_name
    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + "/" + column + "_progression.pdf")
    # only works in python<=3.10
    if save_tikz:
        import tikzplotlib
        tikzplotlib_fix_ncols(plt.gcf())
        tikzplotlib.save(save_path + "/" + column + "_qualitative_analysis.tikz")
    if show:
        plt.show()
