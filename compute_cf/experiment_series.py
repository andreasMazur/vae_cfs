from classifier.clf import train_clf
from compute_cf.compute_cf import compute_counterfactual
from data.data_loading import load_data
from surrogate_model.surrogate_model import train_surrogate
from vae.train_ae import train_vae, normalize_data

from multiprocessing import Pool
from tqdm import tqdm

import numpy as np
import os


def compute_cf_wrapper(Xs, ys, Xs_test, ys_test, vae_path, surrogate_path, N_test=200, max_cf_trials=5, verbose=True):
    _, Xs_means, Xs_stds = normalize_data(Xs)
    Xs_test, ys_test = Xs_test[:N_test], ys_test[:N_test]

    # Try to re-create original W1-values and compare CF to original data point
    cfs = []
    cf_preds = []
    for idx, (target_cf, target_value) in enumerate(zip(Xs_test, ys_test)):
        trial = 0
        restart_cf_run = True
        while restart_cf_run and trial < max_cf_trials:
            if verbose:
                print(f"\nTrial: {trial}")
            config_cf, cf_pred, restart_cf_run = compute_counterfactual(
                Xs,
                ys,
                vae_path,
                surrogate_path,
                target_value=target_value,
                allowed_deviation=0.1,
                allowed_init_deviation=(ys.max() - ys.min()) / 10,
                eta=0.01,
                verbose=verbose,
                restart_if_necessary=trial + 1 < max_cf_trials,
            )
            trial += 1
        cfs.append(config_cf)
        cf_preds.append(cf_pred)
    return np.array(cfs), np.array(cf_preds)


def remove_k_nearest_neighbors(Xs, ys, sample, k):
    # Filter out k nearest neighbors of sample-y-value
    indices = np.abs(ys - sample)[:, 0].argsort()
    remove = indices[:k]
    keep = indices[k:]
    return Xs[keep], ys[keep], Xs[remove], ys[remove]


def train_on_partial_data_wrapper(data_path, logging_dir, repetitions=100, n_test=200, processes=10):
    os.makedirs(logging_dir, exist_ok=True)

    # Select test data
    _, _, (Xs_test, ys_test) = load_data(data_path, splitted=True)
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


def train_on_partial_data(k, logging_dir, data_path, Xs_test, ys_test):
    # Randomly pick a data point in test set and remove its k nearest neighbors in the training data
    (Xs, ys), (Xs_val, ys_val), _ = load_data(data_path, splitted=True)
    selected_outcome = ys_test[np.random.randint(low=0, high=ys_test.shape[0])]
    Xs, ys, Xs_removed, ys_removed = remove_k_nearest_neighbors(Xs, ys, selected_outcome, k)

    # Train new VAE
    vae_path = f"{logging_dir}/vae_{k}_nn_removed"
    if not os.path.isdir(vae_path):
        training_history = None
        while training_history is None or np.isnan(training_history.history["loss"][-1]):
            training_history = train_vae(Xs, Xs_val, Xs_test, logging_dir=vae_path)
        np.save(f"{vae_path}/Xs_train.npy", Xs)
        np.save(f"{vae_path}/ys_train.npy", ys)

    # Train new surrogate model
    surrogate_path = f"{logging_dir}/surrogate_{k}_nn_removed"
    if not os.path.isdir(surrogate_path):
        train_surrogate(
            Xs, ys, Xs_val, ys_val, Xs_test, ys_test, dimensions=[32, 32], logging_dir=surrogate_path
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
