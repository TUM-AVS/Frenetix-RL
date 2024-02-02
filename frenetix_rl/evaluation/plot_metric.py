__author__ = "Rainer Trauth, Alexander Hobmeier"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Release"

import os

import matplotlib.pyplot as plt
import pandas as pd

mod_path = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))


def plot_metric(df, name, titel, show=False, save=True, save_tikz=False):
    plt.plot(df["Step"], df["Value"], color="#005293")
    plt.title(label=titel)
    plt.xlabel("Steps")

    if save_tikz:
        import tikzplotlib
        tikzplotlib.save(mod_path + "/evaluation/metric_plots/" + name + ".tikz")
    if save:
        plt.savefig(mod_path + "/evaluation/metric_plots/" + name + ".pdf")
    if show:
        plt.show()

    plt.clf()


if __name__ == '__main__':
    path = "/media/alex/Windows-SSD/Uni/8. Semester/metrics/"

    name = ["approx_kl", "clip_fraction", "clip_range", "lr", "entropy", "eval_reward", "loss", "std", "value_loss", "variance", "gradient_loss"]
    titel = ['Approximate KL Divergence', "Clip Fraction", "Clip Range", "Learning Rate", "Entropy", "Evaluation Reward", "Training Loss", "Standard Deviation", "Value Loss", "Explained Variance", "Policy Gradient Loss"]

    for number in range(len(name)):
        df = pd.read_csv(path+name[number]+".csv")
        plot_metric(df, name[number], titel[number])