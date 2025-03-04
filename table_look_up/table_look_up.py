from matplotlib import pyplot as plt

import numpy as np
import os


def table_look_up(configurations, angle_outcomes, target_angle):
    closest_outcome_idx = np.abs(angle_outcomes - target_angle).argmin()
    return configurations[closest_outcome_idx], angle_outcomes[closest_outcome_idx]


def table_look_up_experiment(experiment_path, plot=True):
    # Average plot-values over repetitions
    average_table_plot_values, average_cf_plot_values = [], []
    for repetition in [d for d in os.listdir(experiment_path) if "repetition_" in d]:

        # Plot absolute difference vs. k
        table_plot_values, cf_plot_values = [], []
        for k in [4681 * i for i in range(0, 20)]:
            folder_name = f"{experiment_path}/{repetition}/vae_{k}_nn_removed"

            # Load data
            try:
                Xs = np.load(f"{folder_name}/Xs_train.npy")
                ys = np.load(f"{folder_name}/ys_train.npy")
                Xs_test = np.load(f"{folder_name}/targets.npy")
                ys_test = np.load(f"{folder_name}/y_targets.npy")
                cf_preds = np.load(f"{folder_name}/cf_preds.npy")
            except FileNotFoundError:
                print(f"Data misses in: {folder_name}")

            ############################################################
            # Calculate absolute differences for table look-up approach
            ############################################################
            # Average over all targets
            absolute_differences = []
            for target_conf, target_outcome in zip(Xs_test, ys_test):
                # Look up closest configuration in training data
                conf, outcome = table_look_up(Xs, ys, target_angle=target_outcome)

                # Calculate absolute difference
                absolute_difference = np.abs(outcome - target_outcome)
                absolute_differences.append(absolute_difference)

            # Remember plot-values
            table_plot_values.append((k, np.mean(absolute_differences)))

            #######################################################
            # Calculate absolute differences for table CF approach
            #######################################################
            absolute_differences = []
            for target_outcome, cf_outcome in zip(ys_test, cf_preds):
                # Calculate absolute difference
                absolute_difference = np.abs(cf_outcome - target_outcome)
                absolute_differences.append(absolute_difference)
            cf_plot_values.append((k, np.mean(absolute_differences)))

        # Remember repetition plot-values
        average_table_plot_values.append(table_plot_values)
        average_cf_plot_values.append(cf_plot_values)

    # Denominator for normalization
    max_min_distance = ys_test.max() - ys_test.min()

    # Average over table repetitions
    average_table_plot_values = np.array(average_table_plot_values)
    average_table_plot_values[..., 1] = average_table_plot_values[..., 1] / max_min_distance
    table_plot_values_stds = average_table_plot_values.std(axis=0)[:, 1]
    average_table_plot_values = average_table_plot_values.mean(axis=0)

    # Average over cf repetitions
    average_cf_plot_values = np.array(average_cf_plot_values)
    average_cf_plot_values[..., 1] = average_cf_plot_values[..., 1] / max_min_distance
    cf_plot_values_stds = average_cf_plot_values.std(axis=0)[:, 1]
    average_cf_plot_values = average_cf_plot_values.mean(axis=0)

    if plot:
        # Table graph
        plt.plot(average_table_plot_values[:, 0], average_table_plot_values[:, 1], color="blue")
        plt.errorbar(
            average_table_plot_values[:, 0],
            average_table_plot_values[:, 1],
            fmt="none",
            capsize=5,
            yerr=table_plot_values_stds,
            color="blue"
        )
        # CF graph
        plt.plot(average_cf_plot_values[:, 0], average_cf_plot_values[:, 1], color="red")
        plt.errorbar(
            average_cf_plot_values[:, 0],
            average_cf_plot_values[:, 1],
            fmt="none",
            capsize=5,
            yerr=cf_plot_values_stds,
            color="red"
        )

        plt.title("Table look up vs. Counterfactuals")
        plt.xlabel("k-nearest neighbors removed")
        plt.ylabel("mean absolute difference")
        plt.grid()
        plt.show()
    return average_table_plot_values
