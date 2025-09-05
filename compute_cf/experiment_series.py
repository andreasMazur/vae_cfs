from classifier.clf import train_clf
from compute_cf.compute_cf import compute_cf_wrapper
from data.data_loading import load_data
from surrogate_model.surrogate_model import train_surrogate
from vae.train_ae import train_vae

from multiprocessing import Pool
from tqdm import tqdm

import numpy as np
import os


def remove_k_nearest_neighbors(Xs, ys, sample, k):
    """Filter out k nearest neighbors of sample-y-value

    Parameters
    ----------
    Xs: np.ndarray
        The data features.
    ys: np.ndarray
        The data targets.
    sample: np.ndarray
        Selected test sample to remove the neighbors of.
    k: int
        The number of neighbors to remove.
    """
    indices = np.abs(ys - sample)[:, 0].argsort()
    remove = indices[:k]
    keep = indices[k:]
    return Xs[keep], ys[keep], Xs[remove], ys[remove]


def train_on_partial_data(k, logging_dir, data_path, Xs_test, ys_test):
    """ Train models and compute counterfactuals after removing k nearest neighbors of a random test point.

    Parameters
    ----------
    k : int
        Number of nearest neighbors to remove.
    logging_dir: str
        Directory to save logs and models.
    data_path: str
        Path to the CSV file containing the data.
    Xs_test: np.ndarray
        The test data features.
    ys_test: np.ndarray
        The test data targets.
    """
    # Randomly pick a data point in test set and remove its k nearest neighbors in the training data
    (Xs, ys), (Xs_val, ys_val), _ = load_data(data_path, split=True)
    selected_outcome = ys_test[np.random.randint(low=0, high=ys_test.shape[0])]
    Xs, ys, Xs_removed, ys_removed = remove_k_nearest_neighbors(Xs, ys, selected_outcome, k)

    # Train new VAE
    vae_path = f"{logging_dir}/vae_{k}_nn_removed"
    if not os.path.isdir(vae_path):
        training_history = None
        while training_history is None or np.isnan(training_history.history["loss"][-1]):
            training_history = train_vae(Xs, Xs_val, logging_dir=vae_path)
        np.save(f"{vae_path}/Xs_train.npy", Xs)
        np.save(f"{vae_path}/ys_train.npy", ys)

    # Train new surrogate model
    surrogate_path = f"{logging_dir}/surrogate_{k}_nn_removed"
    if not os.path.isdir(surrogate_path):
        train_surrogate(
            Xs, ys, Xs_val, ys_val, mlp_layer_dims=[32, 32], logging_dir=surrogate_path
        )

    # Compute counterfactuals
    cfs_file = f"{vae_path}/cfs.npy"
    cf_preds_file = f"{vae_path}/cf_preds.npy"
    targets_file = f"{vae_path}/targets.npy"
    y_targets_file = f"{vae_path}/y_targets.npy"
    if not os.path.isfile(cfs_file):
        cfs, cf_preds = compute_cf_wrapper(Xs, ys, Xs_test, ys_test, vae_path, surrogate_path, verbose=False)
        np.save(cfs_file, cfs)
        np.save(cf_preds_file, cf_preds)
        np.save(targets_file, Xs_test)
        np.save(y_targets_file, ys_test)

    # Train classifier
    classifier_path = f"{logging_dir}/classifier_{k}_nn_removed"
    if not os.path.isfile(cfs_file):
        train_clf(Xs, Xs_removed, classifier_path)


def train_on_partial_data_wrapper(data_path, logging_dir, repetitions=100, n_test=200, processes=10):
    """ Wrapper to run the experiment series with multiprocessing.

    Parameters
    ----------
    data_path : str
        Path to the CSV file containing the data.
    logging_dir : str
        Directory to save logs and models.
    repetitions : int
        Number of repetitions for the experiment.
    n_test: int
        Number of test samples to consider.
    processes : int
        Number of parallel processes to use.
    """
    os.makedirs(logging_dir, exist_ok=True)

    # Select test data
    _, _, (Xs_test, ys_test) = load_data(data_path, split=True)
    Xs_test, ys_test = Xs_test[:n_test], ys_test[:n_test]

    # Run experiments
    for rep in range(repetitions):
        logging_dir = f"{logging_dir}/repetition_{rep}"
        with Pool(processes) as p:
            p.starmap(
                train_on_partial_data,
                tqdm(
                    [(k, logging_dir, data_path, Xs_test, ys_test) for k in [4681 * i for i in range(8, 18)][::-1]],
                    postfix=f"Currently in repetition {rep}"
                )
            )
