from compute_cf.compute_cf import compute_counterfactual
from data.data_loading import load_data
from surrogate_model.surrogat_model import train_surrogate
from vae.train_ae import train_vae, normalize_data

import numpy as np
import os


def evaluate_cf_quality(Xs, ys, Xs_test, ys_test, vae_path, surrogate_path, N_test=250):
    _, Xs_means, Xs_stds = normalize_data(Xs)
    Xs = Xs[:, [2, 3, 4]]
    Xs_test, ys_test = Xs_test[:N_test], ys_test[:N_test]

    # Try to re-create original W1-values and compare CF to original data point
    cfs = []
    for idx, (target_cf, target_value) in enumerate(zip(Xs_test, ys_test)):
        print(f"\n\nCurrently computing CF {idx} of {Xs_test.shape[0] - 1}")
        print(f"Target CF: {target_cf.tolist()}")
        config_cf, cf_pred = compute_counterfactual(
            Xs,
            ys,
            vae_path,
            surrogate_path,
            (target_cf[:2] - Xs_means[:2]) / Xs_stds[:2],
            target_value,
            allowed_deviation=0.1,
            eta=0.01
        )
        cfs.append(config_cf)
    return np.array(cfs)


def remove_k_nearest_neighbors(Xs, ys, sample, k):
    # Filter for clamping angle and radius
    keep = np.linalg.norm(Xs[:, [3, 4]] - sample[None, [3, 4]], axis=-1).argsort()[k:]
    return Xs[keep], ys[keep]


def train_on_partial_data(path, logging_dir, total_repetitions=5):
    for repetition in range(total_repetitions):
        ks = [1_000, 2_500, 5_000, 7_500, 10_000, 15_000, 20_000, 25_000, 50_000, 100_000]
        for k in ks:
            # Remove k nearest neighbors
            (Xs, ys), (Xs_val, ys_val), (Xs_test, ys_test) = load_data(path, splitted=True)
            # Randomly pick a data point in Xs
            selected_sample = Xs[np.random.randint(low=0, high=Xs.shape[0])]
            Xs, ys = remove_k_nearest_neighbors(Xs, ys, selected_sample, k)

            # - Train new VAE
            print(f"Repetition {repetition}: Now VAE with {k} samples less")
            vae_path = f"{logging_dir}/rep_{repetition}_vae_{k}_nn_removed"
            if not os.path.isdir(vae_path):
                train_vae(Xs, Xs_val, Xs_test, logging_dir=vae_path)
                np.save(f"{vae_path}/Xs_train.npy", Xs)
                np.save(f"{vae_path}/ys_train.npy", ys)

            # - Train new surrogate model
            print(f"Repetition {repetition}: Now surrogate with {k} samples less")
            surrogate_path = f"{logging_dir}/rep_{repetition}_surrogate_{k}_nn_removed"
            if not os.path.isdir(surrogate_path):
                train_surrogate(
                    Xs, ys, Xs_val, ys_val, Xs_test, ys_test, dimensions=[32, 32], logging_dir=surrogate_path
                )

            # - Check quality of counterfactuals
            cfs_file = f"{vae_path}/rep_{repetition}_cfs_{k}_removed.npy"
            targets_file = f"{vae_path}/rep_{repetition}_targets_{k}_removed.npy"
            y_targets_file = f"{vae_path}/rep_{repetition}_y_targets_{k}_removed.npy"
            if not os.path.isfile(cfs_file):
                cfs, targets = evaluate_cf_quality(Xs, ys, Xs_test, ys_test, vae_path, surrogate_path)
                np.save(cfs_file, cfs)
                np.save(targets_file, Xs_test)
                np.save(y_targets_file, ys_test)
