from data.data_loading import load_data
from evaluation.mrad_eval import get_most_similar_config, N_TRAINING_SAMPLES
from vae.train_ae import normalize_data

from tqdm import tqdm
from matplotlib import pyplot as plt

import os
import numpy as np


def compare_outcome_targets(logs_dir, ks, file_name="", max_reps=100):
    """Compute how far off the counterfactual predictions are from the target outcomes.

    Parameters
    ----------
    logs_dir: str
        The directory where the logs from the experiments are stored.
    ks: list
        The list of k values (amount of removed nearest neighbors) to evaluate.
    file_name: str
        The name of the file to store the results in (without extension).
    max_reps: int
        The maximum amount of repetitions to consider.
    """
    rep_dirs = [d for d in os.listdir(logs_dir) if "repetition_" in d]
    rep_dirs.sort(key=lambda path: int(path.split("_")[1]))

    # Average MRAD scores over repetitions
    maes = []
    for repetition in tqdm(rep_dirs[:max_reps]):
        mae_at_k = []
        # Compare MAE for each k
        for removed in ks:
            cfs_y = np.load(f"{logs_dir}/{repetition}/vae_{removed}_nn_removed/cf_preds.npy")
            compare_y = np.load(f"{logs_dir}/{repetition}/vae_{removed}_nn_removed/y_targets.npy")
            mean_absolute_error = np.abs(cfs_y - compare_y).mean()
            mae_at_k.append(((N_TRAINING_SAMPLES - removed) * 100 / N_TRAINING_SAMPLES, mean_absolute_error))
        maes.append(mae_at_k)

    # Store averaged MAE scores
    maes = np.array(maes)
    std_maes = maes[..., 1].std(axis=0)[:, None]
    mean_maes = maes.mean(axis=0)
    result_array = np.concatenate([mean_maes, std_maes], axis=1)
    np.savetxt(
        f"{logs_dir}/{file_name}.csv",
        result_array,
        delimiter=",",
        fmt="%f"
    )

    # Plot
    plt.plot(mean_maes[:, 0], mean_maes[:, 1], color="red")
    plt.errorbar(
        result_array[:, 0],
        result_array[:, 1],
        fmt="none",
        capsize=5,
        yerr=result_array[:, 2],
        color="red"
    )
    plt.title(file_name)
    plt.xlabel("available amount of data")
    plt.ylabel("mean absolute difference")
    plt.grid()
    plt.legend()
    plt.show()


def compare_outcome_closest_nn(data_path, logs_dir, ks, file_name="", max_reps=100):
    """Compute how far off the counterfactual predictions are from their nearest neighbor outcome bending angles.

    Parameters
    ----------
    data_path: str
        The path to the data file.
    logs_dir: str
        The directory where the logs from the experiments are stored.
    ks: list
        The list of k values (amount of removed nearest neighbors) to evaluate.
    file_name: str
        The name of the file to store the results in (without extension).
    max_reps: int
        The maximum amount of repetitions to consider.
    """
    # Load and normalize data
    Xs, ys = load_data(data_path, split=False)
    Xs, Xs_means, Xs_stds = normalize_data(Xs)

    rep_dirs = [d for d in os.listdir(logs_dir) if "repetition_" in d]
    rep_dirs.sort(key=lambda path: int(path.split("_")[1]))

    # Average MRAD scores over repetitions
    maes = []
    for repetition in tqdm(rep_dirs[:max_reps]):
        mae_at_k = []
        # Compare MAE for each k
        for removed in ks:
            cfs = np.load(f"{logs_dir}/{repetition}/vae_{removed}_nn_removed/cfs.npy")
            cfs_y = np.load(f"{logs_dir}/{repetition}/vae_{removed}_nn_removed/cf_preds.npy")

            # Find nearest-neighbor to (normalized) counterfactual
            cfs = (cfs - Xs_means) / Xs_stds
            _, closest_y = get_most_similar_config(Xs, ys, cfs)

            mean_absolute_error = np.abs(cfs_y - closest_y).mean()
            mae_at_k.append(((N_TRAINING_SAMPLES - removed) * 100 / N_TRAINING_SAMPLES, mean_absolute_error))
        maes.append(mae_at_k)

    # Store averaged MAE scores
    maes = np.array(maes)
    std_maes = maes[..., 1].std(axis=0)[:, None]
    mean_maes = maes.mean(axis=0)
    result_array = np.concatenate([mean_maes, std_maes], axis=1)
    np.savetxt(
        f"{logs_dir}/{file_name}.csv",
        result_array,
        delimiter=",",
        fmt="%f"
    )

    # Plot
    plt.plot(mean_maes[:, 0], mean_maes[:, 1], color="red")
    plt.errorbar(
        result_array[:, 0],
        result_array[:, 1],
        fmt="none",
        capsize=5,
        yerr=result_array[:, 2],
        color="red"
    )
    plt.title(file_name)
    plt.xlabel("available amount of data")
    plt.ylabel("mean absolute difference")
    plt.grid()
    plt.legend()
    plt.show()
