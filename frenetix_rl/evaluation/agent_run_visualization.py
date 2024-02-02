__author__ = "Rainer Trauth, Alexander Hobmeier"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Release"

import os
import pandas as pd
import matplotlib.pyplot as plt
import yaml


def load_yaml_file(file_path):
    """
    Load a YAML file and return its contents.

    :param file_path: Path to the YAML file.
    :return: Data from the YAML file.
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def create_cumulative_sum_column(df, column_name, initial_value):
    """
    Create a new column where each row is the cumulative sum of the specified column,
    starting with an initial value.

    :param df: pandas DataFrame containing the data.
    :param column_name: Name of the column for cumulative sum.
    :param initial_value: The initial value to start the cumulative sum.
    :return: DataFrame with the new cumulative sum column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    # Cumulative sum of the specified column
    df[f'{column_name}_cumulative'] = df[column_name].cumsum() + initial_value

    return df


def plot_columns_vs_s_position_actions(df, columns_to_plot, save_path, mod_path):
    """
    Plots selected columns from a DataFrame against the 's_position_m' column.

    :param df: pandas DataFrame containing the data.
    :param columns_to_plot: List of column names to plot against 's_position_m'.
    """

    # Load Cost params
    cost_weights = load_yaml_file(os.path.join(mod_path, "configurations", "frenetix_motion_planner", "cost.yaml"))["cost_weights"]

    # Check if 's_position_m' is in the dataframe
    if 's_position_m' not in df:
        raise ValueError("DataFrame must contain a 's_position_m' column")

    # Create a figure and axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot each column against 's_position_m'
    for col in columns_to_plot:
        if col in df:
            name = next((key for key in cost_weights.keys() if key.startswith(col[:10])), None)
            if name is not None:
                df = create_cumulative_sum_column(df, col, cost_weights[name])
                ax1.plot(df['s_position_m'], df[col], label=col)
                ax2.plot(df['s_position_m'], df[f'{col}_cumulative'], label=f'{col}_cumulative')
            else:
                print(f"No matching cost weight found for column '{col}'")
        else:
            print(f"Column '{col}' not found in DataFrame.")

    # Setting the plot titles and labels for the first subplot
    ax1.set_title('Original Columns Plotted Against s_position_m')
    ax1.set_xlabel('s_position_m')
    ax1.set_ylabel('Values')
    ax1.legend()

    # Setting the plot titles and labels for the second subplot
    ax2.set_title('Cumulative Columns Plotted Against s_position_m')
    ax2.set_xlabel('s_position_m')
    ax2.set_ylabel('Cumulative Values')
    ax2.legend()

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    # plt.show()
    plt.savefig(os.path.join(save_path, "analyzing_plot.svg"), format='svg')
    plt.close()


def visualize_agent_run(log_path, mod_path, planner_logs_path: str = None, agent_logs_path: str = None):

    # Assuming the file paths
    agent_logs_path = agent_logs_path or os.path.join(log_path, "agent_logs.csv")
    planner_logs_path = planner_logs_path or os.path.join(log_path, "logs.csv")

    # Loading the log files
    agent_logs = pd.read_csv(agent_logs_path, sep=";")
    planner_logs = pd.read_csv(planner_logs_path, sep=";")

    min_length = min(len(agent_logs), len(planner_logs))
    agent_logs = agent_logs.head(min_length)
    planner_logs = planner_logs.head(min_length)

    data = pd.concat([agent_logs, planner_logs], axis=1)
    s_position_progress = data['s_position_m'] - data['s_position_m'].iloc[0]
    data.insert(0, 's_position_progress', s_position_progress)
    plot_columns_vs_s_position_actions(data, ['prediction_action'], log_path, mod_path)
    print("DONE")


# Testing Purposes
if __name__ == '__main__':
    mod_path = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))
    log_path_ = os.path.join(mod_path, "logs", "60000", "0")
    visualize_agent_run(log_path_, mod_path)
