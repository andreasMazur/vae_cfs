from matplotlib import pyplot as plt

import os
import numpy as np

FEATURE_NAMES = {
    0: "e_modulus",
    1: "wire_thickness",
    2: "bending_angle",
    3: "bending_radius"
}
N_TRAINING_SAMPLES = 93_627


def cf_evaluation(logs_dir, plot=True, max_removed=20):
    for feature in range(4):
        plot_values = []
        for repetition in [d for d in os.listdir(logs_dir) if "repetition_" in d][:1]:
            repetition_plot_values = []
            for removed in [4681 * i for i in range(0, 20)][::-1][:max_removed]:
                # Calculate mean absolute relative difference
                cfs = np.load(f"{logs_dir}/{repetition}/vae_{removed}_nn_removed/cfs.npy")[:, feature]
                targets = np.load(
                    f"{logs_dir}/{repetition}/vae_{removed}_nn_removed/targets.npy"
                )[:, feature]
                total_feature_difference = np.abs(cfs - targets).mean()
                repetition_plot_values.append((N_TRAINING_SAMPLES - removed, total_feature_difference))
            plot_values.append(repetition_plot_values)

        # Compute mean over repetitions
        plot_values = np.array(plot_values)
        plot_values[:, :, 1] = plot_values[:, :, 1] / (targets.max() - targets.min())

        # Save mean-values and std-values
        plot_means = plot_values.mean(axis=0)
        plt_std = plot_values.std(axis=0)[..., 1]
        np.savetxt(f"{logs_dir}/mean_values_{FEATURE_NAMES[feature]}.csv", plot_means, delimiter=",", fmt="%f")
        np.savetxt(f"{logs_dir}/std_values_{FEATURE_NAMES[feature]}.csv", plt_std, delimiter=",", fmt="%f")

        if plot:
            plt.plot(plot_means[:, 0], plot_means[:, 1], color="blue")
            plt.errorbar(plot_means[:, 0], plot_means[:, 1], fmt="none", capsize=5, yerr=plt_std, color="blue")
            plt.title(f"Mean over absolute errors: {FEATURE_NAMES[feature]}")
            plt.grid()
            plt.show()
