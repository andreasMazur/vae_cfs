from evaluation.mrad_eval import N_TRAINING_SAMPLES

from tqdm import tqdm
from matplotlib import pyplot as plt

import os
import numpy as np


def table_look_up(configurations, angle_outcomes, target_angle):
    """Look up the closest configuration in the training data and return it and its outcome.

    Parameters
    ----------
    configurations: np.ndarray
        The process configurations from the dataset that have been used for training.
    angle_outcomes: np.ndarray
        The corresponding angle outcomes to the process configurations from the dataset that have been used for
        training.
    target_angle: float
        The target angle outcome that is supposed to be achieved.

    Returns
    -------
    (np.ndarray, float)
        The closest process configuration and its corresponding outcome.
    """
    closest_outcome_idx = np.abs(angle_outcomes - target_angle).argmin()
    return configurations[closest_outcome_idx], angle_outcomes[closest_outcome_idx]


def baseline_performance(logs_dir, ks, file_name, max_reps=100):
    """Determine for different k how far off the table look-up approach is from the target outcomes.

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

    maes = []
    for repetition in tqdm(rep_dirs[:max_reps]):
        # Compute MRAD for each k
        maes_at_k = []
        for removed in ks:
            # Load training data and y-targets
            Xs = np.load(f"{logs_dir}/{repetition}/vae_{removed}_nn_removed/Xs_train.npy")
            ys = np.load(f"{logs_dir}/{repetition}/vae_{removed}_nn_removed/ys_train.npy")
            y_targets = np.load(f"{logs_dir}/{repetition}/vae_{removed}_nn_removed/y_targets.npy")

            # Compute absolute differences for table look-up approach
            aes_at_k = []
            for y_target in y_targets:
                # Look up closest configuration in training data
                _, outcome = table_look_up(Xs, ys, target_angle=y_target)

                # Calculate absolute difference
                ae = np.abs(outcome - y_target)
                aes_at_k.append(ae)

            # Add mean absolute difference to list over all k
            maes_at_k.append(((N_TRAINING_SAMPLES - removed) * 100 / N_TRAINING_SAMPLES, np.mean(aes_at_k)))
        maes.append(maes_at_k)

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
