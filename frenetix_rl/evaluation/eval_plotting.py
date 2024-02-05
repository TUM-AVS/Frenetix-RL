__author__ = "Rainer Trauth, Alexander Hobmeier"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Release"

import os
import numpy as np
from frenetix_rl.evaluation.qualitative_plotting import tikzplotlib_fix_ncols
from matplotlib import pyplot as plt

mod_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def box_plot(dfs, column, show=False, showoutlier=False, save=True, save_tikz=False):
    plt.rcParams.update({'font.size': 15})
    data = []
    keys = []
    for key, df in dfs.items():
        data.append(np.array(df[column]))
        keys.append(key)

    # plotting params
    colors = ["#003359", "#005293", "#98C6EA", "#808080", "#E37222"]
    colors = colors[1:]
    meanpointprops = dict(marker='x', markeredgecolor='black', markerfacecolor='black')
    medianprops = dict(color="black")

    # plot
    fig, ax = plt.subplots(figsize=(9, 7))
    bplot = ax.boxplot(data, showmeans=True, meanprops=meanpointprops, showfliers=showoutlier, patch_artist=True, medianprops=medianprops)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    ax.set(xticklabels=[])
    ax.set(xlabel=None)

    # build legend
    handles = []
    for i in range(len(data)):
        import matplotlib.patches as mpatches
        handles.append(mpatches.Patch(color=colors[i], label=keys[i]))
    ax.legend(keys, handles=handles, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
              mode="expand", borderaxespad=0, ncol=3, frameon=False)
    plt.ylabel(column.replace("_", " ").title().replace("Obst", "Obstacle").replace("Std", "Standard Deviation"))

    # save figures
    if save:
        plt.savefig(mod_path + "/evaluation/plots/quantitative_plots/"+column+"_boxplot.pdf")
    # only works in python<=3.10
    if save_tikz:
        import tikzplotlib
        tikzplotlib_fix_ncols(fig)
        tikzplotlib.save(mod_path + "/evaluation/plots/quantitative_plots/"+column+"_boxplot.tikz")
    if show:
        plt.show()


def box_plot_risk(dfs, column, show=False, showoutlier=False, save=True, save_tikz=False):
    plt.rcParams.update({'font.size': 15})
    data = []
    keys = []
    for key, df in dfs.items():
        data.append(np.array(df[column]))
        keys.append(key)
    for key, df in dfs.items():
        data.append(np.array(df[column.replace("ego", "obst")]))

    # plotting params
    colors = ["#005293", "#E37222"]
    meanpointprops = dict(marker='x', markeredgecolor='black', markerfacecolor='black')
    medianprops = dict(color="black")

    # plot
    fig, ax = plt.subplots(figsize=(9, 7))
    bplot = ax.boxplot(data, showmeans=True, meanprops=meanpointprops, showfliers=showoutlier, patch_artist=True, medianprops=medianprops)
    for i, patch in enumerate(bplot['boxes']):
        patch.set_facecolor(colors[int(i % len(colors))])
    ax.set(xticklabels=["                         Ego", "", "                         3rd party", ""])
    ax.set(xlabel=None)

    # build legend
    handles = []
    for i in range(len(keys)):
        import matplotlib.patches as mpatches
        handles.append(mpatches.Patch(color=colors[int(i % len(colors))], label=keys[int(i % len(keys))]))
    ax.legend(keys, handles=handles, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3, frameon=False)
    plt.ylabel(column.replace("obst", "").replace("ego", "").replace("__", "_").replace("_", " ").title().replace("Std", "Standard Deviation"))

    plt.gcf().subplots_adjust(left=0.2)

    # save figures
    if save:
        plt.savefig(mod_path + "/evaluation/plots/quantitative_plots/"+column.replace("_ego", "")+"_boxplot.pdf")
    # only works in python<=3.10
    if save_tikz:
        import tikzplotlib
        tikzplotlib_fix_ncols(fig)
        tikzplotlib.save(mod_path + "/evaluation/plots/quantitative_plots/"+column.replace("_ego", "")+"_boxplot.tikz")
    if show:
        plt.show()


def box_plot_harm(dfs, show=False, showoutlier=False, save=True, save_tikz=False):
    plt.rcParams.update({'font.size': 15})
    for column in ["ego_harm", "obst_harm"]:
        data = []
        keys = []
        for key, df in dfs.items():
            harm_data = np.array(df[column])
            harm_data = harm_data[~np.isnan(harm_data)]
            harm_data = harm_data[harm_data != 0]
            data.append(harm_data)
            keys.append(key)

        # plotting params
        colors = ["#003359", "#005293", "#98C6EA", "#808080", "#E37222"]
        colors = colors[1:]
        meanpointprops = dict(marker='x', markeredgecolor='black', markerfacecolor='black')
        medianprops = dict(color="black")

        # plot
        fig, ax = plt.subplots(figsize=(9, 7))
        bplot = ax.boxplot(data, showmeans=True, meanprops=meanpointprops, showfliers=showoutlier,
                           patch_artist=True, medianprops=medianprops)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        ax.set(xticklabels=[])
        ax.set(xlabel=None)

        # build legend
        handles = []
        for i in range(len(data)):
            import matplotlib.patches as mpatches
            handles.append(mpatches.Patch(color=colors[i], label=keys[i]))
        ax.legend(keys, handles=handles, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                  mode="expand", borderaxespad=0, ncol=3, frameon=False)
        plt.ylabel(
            column.replace("_", " ").title().replace("Obst", "Obstacle").replace("Std", "Standard Deviation"))

        # save figures
        if save:
            plt.savefig(mod_path + "/evaluation/plots/quantitative_plots/" + column + "_boxplot.pdf")
        # only works in python<=3.10
        if save_tikz:
            import tikzplotlib
            tikzplotlib_fix_ncols(fig)
            tikzplotlib.save(mod_path + "/evaluation/plots/quantitative_plots/" + column + "_boxplot.tikz")
        if show:
            plt.show()


def success_pie_plot(dfs, show=False, save=True, save_tikz=False):
    data = []
    keys = []
    for key, df in dfs.items():
        data.append(np.array(df["message"]))
        keys.append(key)


    fig, axes = plt.subplots(1, 2, figsize=(8,4))
    colors = {"Success": "#005293", "Failed": "#98C6EA"}
    label_names = {"Success": "Success", "Failed": "Failure"}
    plt.suptitle("Success rates")

    for i, ax in enumerate(axes):
        label, counts = np.unique(data[i], return_counts=True)
        ax.pie(counts, labels=[label_names[l] for l in label], colors=[colors[l] for l in label], autopct='%1.1f%%')
        ax.set_title(keys[i])

    # save figures
    if save:
        plt.savefig(mod_path + "/evaluation/plots/quantitative_plots/success_pie.pdf")
    # only works in python<=3.10
    if save_tikz:
        import tikzplotlib
        tikzplotlib_fix_ncols(fig)
        tikzplotlib.save(mod_path + "/evaluation/plots/quantitative_plots/success_pie.tikz")
    if show:
        plt.show()

