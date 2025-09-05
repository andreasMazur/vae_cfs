from data.data_loading import load_data
from vae.train_ae import normalize_data, de_normalize_data

from tqdm import tqdm
from matplotlib import pyplot as plt

import numpy as np
import os
import json

FEATURE_NAMES = {
    0: "e_modulus",
    1: "wire_thickness",
    2: "bending_angle",
    3: "bending_radius"
}
N_TRAINING_SAMPLES = 93_627


def mrad_eval_target(data_path, logs_dir, data_distance_path, ks, feature, file_name="", max_reps=100):
    """Create MRAD plots over different amounts of training data.

    Parameters
    ----------
    data_path: str
        Path to the data folder containing the dataset.
    logs_dir: str
        The root directory where the logs of the experiments are stored.
    data_distance_path: str
        The path to the JSON file containing intra-dataset total feature distances.
    ks: list
        A list of integers representing the number of removed training samples.
    feature: int
        The index of the feature for which the MRAD is to be calculated.
    file_name: str
        The name of the file where the plot values will be saved (without extension).
    max_reps: int
        The maximum number of repetitions to consider for averaging.
    """
    # Load the data
    Xs, ys = load_data(data_path, split=False)

    # Compute MRAD denominator
    mrad_denominator = Xs[:, feature].max() - Xs[:, feature].min()

    # Setup iteration over repetitions
    rep_dirs = [d for d in os.listdir(logs_dir) if "repetition_" in d]
    rep_dirs.sort(key=lambda path: int(path.split("_")[1]))
    mean_mrads = []

    # Load intra-dataset feature distance
    d = json.load(open(data_distance_path))

    # Average MRAD scores over repetitions
    for repetition in tqdm(rep_dirs[:max_reps]):
        mrad_at_k = []
        # Compute MRAD for each k
        for removed in ks:
            cfs = np.load(f"{logs_dir}/{repetition}/vae_{removed}_nn_removed/cfs.npy")[:, feature]
            compare_feature = np.load(f"{logs_dir}/{repetition}/vae_{removed}_nn_removed/targets.npy")[:, feature]
            mrad = np.abs(cfs - compare_feature).mean() / mrad_denominator
            mrad_at_k.append(
                ((N_TRAINING_SAMPLES - removed) * 100 / N_TRAINING_SAMPLES, mrad - d[FEATURE_NAMES[feature]])
            )
        mean_mrads.append(mrad_at_k)

    # Store averaged MRAD scores
    mrads = np.array(mean_mrads)
    mean_mrads = mrads.mean(axis=0)
    std_mrads = mrads[..., 1].std(axis=0)
    np.savetxt(
        f"{logs_dir}/{file_name}.csv",
        mean_mrads,
        delimiter=",",
        fmt="%f"
    )

    # Plot averaged MRAD scores
    plt.plot(mean_mrads[:, 0], mean_mrads[:, 1], color="blue")
    plt.errorbar(mean_mrads[:, 0], mean_mrads[:, 1], fmt="none", capsize=5, yerr=std_mrads, color="blue")
    plt.title(f"Mean absolute error: {FEATURE_NAMES[feature]}")
    plt.grid()
    plt.show()


def get_most_similar_config(Xs, ys, x_cf):
    """Find the most similar configuration in the dataset to a given counterfactual.

    Parameters
    ----------
    Xs: np.ndarray
        The process configurations contained in the dataset.
    ys: np.ndarray
        The outcome bending angles contained in the dataset.
    x_cf: np.ndarray
        The counterfactual configurations for which the most similar configuration in the dataset is to be found.

    Returns
    -------
    (np.ndarray, np.ndarray)
        The most similar configurations to the counterfactuals and their corresponding outcomes.
    """
    closest_feature_indices = np.linalg.norm(x_cf[:, None, :] - Xs[None, :, :], axis=-1).argmin(axis=-1)
    return Xs[closest_feature_indices], ys[closest_feature_indices]


def mrad_eval_closest_nn(data_path, logs_dir, data_distance_path, ks, feature, file_name="", max_reps=100):
    """Compute MRAD scores of counterfactuals compared to their closest neighbor in the entire dataset.

    Parameters
    ----------
    data_path: str
        Path to the data folder containing the dataset.
    logs_dir: str
        The root directory where the logs of the experiments are stored.
    data_distance_path: str
        The path to the JSON file containing intra-dataset total feature distances.
    ks: list
        A list of integers representing the number of removed training samples.
    feature: int
        The index of the feature for which the MRAD is to be calculated.
    file_name: str
        The name of the file where the plot values will be saved (without extension).
    max_reps: int
        The maximum number of repetitions to consider for averaging.
    """
    # Load and normalize data
    Xs, ys = load_data(data_path, split=False)
    mrad_denominator = Xs[:, feature].max() - Xs[:, feature].min()
    Xs, Xs_means, Xs_stds = normalize_data(Xs)

    # Setup iteration
    rep_dirs = [d for d in os.listdir(logs_dir) if "repetition_" in d]
    rep_dirs.sort(key=lambda path: int(path.split("_")[1]))
    mean_mrads = []

    # Load intra-dataset feature distance
    d = json.load(open(data_distance_path))

    # Average MRAD scores over repetitions
    for repetition in tqdm(rep_dirs[:max_reps]):
        mrad_at_k = []
        # Compute MRAD for each k
        for removed in ks:
            # Load counterfactuals
            cfs = np.load(f"{logs_dir}/{repetition}/vae_{removed}_nn_removed/cfs.npy")
            cfs = (cfs - Xs_means) / Xs_stds

            # Find nearest-neighbor to counterfactual
            closest_x, _ = get_most_similar_config(Xs, ys, cfs)

            # De-normalize data
            closest_x = de_normalize_data(closest_x, Xs_means, Xs_stds)
            cfs = de_normalize_data(cfs, Xs_means, Xs_stds)

            # Select feature of interest
            cfs = cfs[:, feature]
            compare_feature = closest_x[:, feature]

            # Compute MRAD
            mrad = np.abs(cfs - compare_feature).mean() / mrad_denominator
            mrad_at_k.append(
                ((N_TRAINING_SAMPLES - removed) * 100 / N_TRAINING_SAMPLES, mrad - d[FEATURE_NAMES[feature]])
            )
        mean_mrads.append(mrad_at_k)

    # Store averaged MRAD scores
    mrads = np.array(mean_mrads)
    mean_mrads = mrads.mean(axis=0)
    std_mrads = mrads[..., 1].std(axis=0)
    np.savetxt(
        f"{logs_dir}/{file_name}.csv",
        mean_mrads,
        delimiter=",",
        fmt="%f"
    )

    # Plot averaged MRAD scores
    plt.plot(mean_mrads[:, 0], mean_mrads[:, 1], color="blue")
    plt.errorbar(mean_mrads[:, 0], mean_mrads[:, 1], fmt="none", capsize=5, yerr=std_mrads, color="blue")
    plt.title(f"Mean absolute error: {FEATURE_NAMES[feature]}")
    plt.grid()
    plt.show()
