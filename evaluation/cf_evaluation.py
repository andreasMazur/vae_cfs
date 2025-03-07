from data.data_loading import load_data

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


def get_most_similar_config(Xs, ys, x_cf):
    closest_feature_indices = np.linalg.norm(x_cf[:, None, :] - Xs[None, :, :], axis=-1).argmin(axis=-1)
    return Xs[closest_feature_indices], ys[closest_feature_indices]


def cf_evaluation(data_path, logs_dir, plot=True, max_removed=20):
    Xs_all, _ = load_data(data_path, splitted=False)
    (Xs, ys), _, _ = load_data(data_path, splitted=True)
    for feature in range(4):
        gt_plot_values = []
        closest_plot_values = []
        data_misses = False
        for repetition in [d for d in os.listdir(logs_dir) if "repetition_" in d]:
            repetition_gt_plot_values = []
            repetition_closest_plot_values = []
            for removed in [4681 * i for i in range(8, 18)][::-1][:max_removed]:
                ########################################################################################
                # Calculate mean absolute relative difference (w.r.t. true process config for target y)
                ########################################################################################
                try:
                    cfs = np.load(f"{logs_dir}/{repetition}/vae_{removed}_nn_removed/cfs.npy")
                    cfs_feature = cfs[:, feature]
                    true_x_for_target = np.load(
                        f"{logs_dir}/{repetition}/vae_{removed}_nn_removed/targets.npy"
                    )[:, feature]
                    total_feature_difference = np.abs(cfs_feature - true_x_for_target).mean()
                    repetition_gt_plot_values.append(
                        ((N_TRAINING_SAMPLES - removed) * 100 / Xs_all.shape[0], total_feature_difference)
                    )
                except FileNotFoundError:
                    print(f"Data misses in: {logs_dir}/{repetition}/vae_{removed}_nn_removed/")
                    data_misses = True
                    break

                #############################################################################################
                # Calculate mean absolute relative difference (w.r.t. most similar config to counterfactual)
                #############################################################################################
                closest_x, _ = get_most_similar_config(Xs, ys, cfs)
                closest_x_feature = closest_x[:, feature]
                total_feature_difference = np.abs(cfs_feature - closest_x_feature).mean()
                repetition_closest_plot_values.append(
                    ((N_TRAINING_SAMPLES - removed) * 100 / Xs_all.shape[0], total_feature_difference)
                )

            gt_plot_values.append(repetition_gt_plot_values)
            closest_plot_values.append(repetition_closest_plot_values)
        # Don't add repetitions which lack data
        if data_misses:
            continue

        # Compute mean over repetitions
        gt_plot_values = np.array(gt_plot_values)
        gt_plot_values[:, :, 1] = gt_plot_values[:, :, 1] / (true_x_for_target.max() - true_x_for_target.min())
        closest_plot_values = np.array(closest_plot_values)
        closest_plot_values[:, :, 1] = closest_plot_values[:, :, 1] / (Xs[:, feature].max() - Xs[:, feature].min())

        # Save mean-values and std-values
        test_plot_means = gt_plot_values.mean(axis=0)
        test_plt_std = gt_plot_values.std(axis=0)[..., 1]
        np.savetxt(f"{logs_dir}/test_mean_values_{FEATURE_NAMES[feature]}.csv", test_plot_means, delimiter=",", fmt="%f")
        np.savetxt(f"{logs_dir}/test_std_values_{FEATURE_NAMES[feature]}.csv", test_plt_std, delimiter=",", fmt="%f")

        closest_plot_means = closest_plot_values.mean(axis=0)
        closest_plt_std = closest_plot_values.std(axis=0)[..., 1]
        np.savetxt(f"{logs_dir}/closest_mean_values_{FEATURE_NAMES[feature]}.csv", closest_plot_means, delimiter=",", fmt="%f")
        np.savetxt(f"{logs_dir}/closest_std_values_{FEATURE_NAMES[feature]}.csv", closest_plt_std, delimiter=",", fmt="%f")

        if plot:
            plt.plot(test_plot_means[:, 0], test_plot_means[:, 1], color="blue", label="test")
            plt.errorbar(test_plot_means[:, 0], test_plot_means[:, 1], fmt="none", capsize=5, yerr=test_plt_std, color="blue")

            plt.plot(closest_plot_means[:, 0], closest_plot_means[:, 1], color="red", label="closest")
            plt.errorbar(closest_plot_means[:, 0], closest_plot_means[:, 1], fmt="none", capsize=5, yerr=closest_plt_std, color="red")

            plt.title(f"Mean over absolute errors: {FEATURE_NAMES[feature]}")
            plt.grid()
            plt.legend()
            plt.show()
