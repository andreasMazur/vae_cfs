from compute_cf.compute_cf import compute_counterfactual
from data.data_loading import load_data, COL_BENDING_ANGLE, COL_BENDING_RADIUS
from surrogate_model.surrogate_model import train_surrogate
from vae.train_ae import train_vae, normalize_data

from multiprocessing import Pool

import numpy as np
import os


def evaluate_cf_quality(Xs, ys, Xs_test, ys_test, vae_path, surrogate_path, N_test=250):
    _, Xs_means, Xs_stds = normalize_data(Xs)
    Xs = Xs[:, [COL_BENDING_ANGLE, COL_BENDING_RADIUS]]
    Xs_test, ys_test = Xs_test[:N_test], ys_test[:N_test]

    # Try to re-create original W1-values and compare CF to original data point
    cfs = []
    cf_preds = []
    for idx, (target_cf, target_value) in enumerate(zip(Xs_test, ys_test)):
        config_cf, cf_pred = compute_counterfactual(
            Xs,
            ys,
            vae_path,
            surrogate_path,
            material_information=(target_cf[:2] - Xs_means[:2]) / Xs_stds[:2],
            target_value=target_value,
            allowed_deviation=0.1,
            eta=0.01,
            verbose=True
        )
        cfs.append(config_cf)
        cf_preds.append(cf_pred)
    return np.array(cfs), np.array(cf_preds)


def remove_k_nearest_neighbors(Xs, ys, sample, k):
    # Filter for clamping angle and radius
    neighbors = Xs[:, [COL_BENDING_ANGLE, COL_BENDING_RADIUS]]
    sample = sample[None, [COL_BENDING_ANGLE, COL_BENDING_RADIUS]]
    keep = np.linalg.norm(neighbors - sample, axis=-1).argsort()[k:]
    return Xs[keep], ys[keep]


def multiprocessing_experiment_series(data_path, logging_dir, repetitions):
    logging_dirs = [f"{logging_dir}/repetition_{i}" for i in range(repetitions)]
    with Pool(repetitions) as p:
        p.starmap(train_on_partial_data, [(data_path, ld) for ld in logging_dirs])


def train_on_partial_data(data_path, logging_dir, n_test=200):
    # Select test data
    _, _, (Xs_test, ys_test) = load_data(data_path, splitted=True)
    Xs_test, ys_test = Xs_test[:n_test], ys_test[:n_test]

    for k in [4681 * i for i in range(0, 20)]:
        # Randomly pick a data point in test set and remove its k nearest neighbors in the training data
        (Xs, ys), (Xs_val, ys_val), _ = load_data(data_path, splitted=True)
        selected_sample = Xs_test[np.random.randint(low=0, high=n_test)]
        Xs, ys = remove_k_nearest_neighbors(Xs, ys, selected_sample, k)

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

        # Check quality of counterfactuals
        cfs_file = f"{vae_path}/cfs.npy"
        cf_preds_file = f"{vae_path}/cf_preds.npy"
        targets_file = f"{vae_path}/targets.npy"
        y_targets_file = f"{vae_path}/y_targets.npy"
        if not os.path.isfile(cfs_file):
            cfs, cf_preds = evaluate_cf_quality(Xs, ys, Xs_test, ys_test, vae_path, surrogate_path)
            np.save(cfs_file, cfs)
            np.save(cf_preds_file, cf_preds)
            np.save(targets_file, Xs_test)
            np.save(y_targets_file, ys_test)
