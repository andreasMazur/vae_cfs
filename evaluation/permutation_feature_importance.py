from data.data_loading import load_data
from evaluation.mrad_eval import N_TRAINING_SAMPLES
from surrogate_model.surrogate_model import r2_score
from vae.train_ae import normalize_data, de_normalize_data

from tqdm import tqdm
from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf
import os


def permutation_feature_importance(surrogate_path, feature_idx, Xs, ys):
    """Calculate the permutation feature importance for a specific feature.

    Parameters
    ----------
    surrogate_path: str
        The path to the trained surrogate model.
    feature_idx: int
        The index of the feature for which the PFI is to be calculated.
    Xs: np.array
        All process configurations from the dataset.
    ys: np.array
        All angle outcomes from the dataset.

    Returns
    -------
    float:
        The permutation feature importance for the specified feature.
    """
    Xs, Xs_means, Xs_std = normalize_data(Xs)
    _, ys_mean, ys_std = normalize_data(ys)
    model = tf.keras.models.load_model(surrogate_path, custom_objects={"r2_score": r2_score})

    # Original score
    y_pred = de_normalize_data(model(Xs), ys_mean, ys_std)
    r2_original = r2_score(ys, y_pred.numpy())

    # Shuffle data
    shuffled_features = Xs[:, feature_idx]
    np.random.shuffle(shuffled_features)
    Xs = np.concatenate([Xs[:, :feature_idx], shuffled_features[:, None], Xs[:, feature_idx+1:]], axis=-1)
    y_pred = de_normalize_data(model(Xs), ys_mean, ys_std)
    r2_shuffled = r2_score(ys, y_pred.numpy())

    # Permutation score
    return np.abs(r2_shuffled - r2_original)


def pfi_plots(data_path, logs_dir, ks, feature, file_name="", max_reps=100):
    """Create permutation feature importance plots for a specific feature over different amounts of training data.

    Parameters
    ----------
    data_path : str
        Path to the data folder containing the dataset.
    logs_dir : str
        The root directory where the logs of the experiments are stored.
    ks: list
        A list of integers representing the number of removed training samples.
    feature: int
        The index of the feature for which the PFI is to be calculated.
    file_name: str
        The name of the file where the plot values will be saved (without extension).
    max_reps: int
        The maximum number of repetitions to consider for averaging.
    """
    # Load all data
    Xs, ys = load_data(data_path, split=False)

    # Prepare directories
    rep_dirs = [d for d in os.listdir(logs_dir) if "repetition_" in d]
    rep_dirs.sort(key=lambda path: int(path.split("_")[1]))

    averaged_pfi = []
    for repetition in tqdm(rep_dirs[:max_reps]):
        feature_pfi = []
        for removed in ks:
            pfi = permutation_feature_importance(
                surrogate_path=f"{logs_dir}/{repetition}/surrogate_{removed}_nn_removed/surrogate.keras",
                feature_idx=feature,
                Xs=Xs,
                ys=ys
            )
            feature_pfi.append(((N_TRAINING_SAMPLES - removed) * 100 / N_TRAINING_SAMPLES, pfi))
        averaged_pfi.append(feature_pfi)

    # Calculate averaged PFI over repetition and corresponding standard deviations
    plot_means = np.array(averaged_pfi)
    plot_means = plot_means.mean(axis=0)

    # Save plot-values
    np.savetxt(
        f"{logs_dir}/{file_name}.csv", plot_means, delimiter=",", fmt="%f"
    )

    # Plot
    plt.tight_layout()
    plt.plot(plot_means[:, 0], plot_means[:, 1], color="blue")
    plt.grid()
    plt.xlabel("Percentage of available data")
    plt.ylabel("Permutation Feature Importance")
    plt.title(f"PFI for feature {feature}")
    plt.show()
