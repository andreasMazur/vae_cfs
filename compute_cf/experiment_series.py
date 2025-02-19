from compute_cf.compute_cf import compute_counterfactual
from data.data_loading import load_data
from surrogate_model.surrogat_model import train_surrogate
from vae.train_ae import train_vae, normalize_data

import numpy as np
import os


def evaluate_cf_quality(Xs, ys, Xs_test, ys_test, vae_path, surrogate_path):
    _, Xs_means, Xs_stds = normalize_data(Xs)
    Xs = Xs[:, [2, 3, 4]]

    # Try to re-create original W1-values and compare CF to original data point
    abs_deviations = []
    for target_cf, target_value in zip(Xs_test, ys_test):
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
        abs_deviations.append(np.abs(config_cf - target_cf[2:]).sum())
    return np.array(abs_deviations)


def remove_k_nearest_neighbors(Xs, ys, sample, k):
    # Filter for clamping angle and radius
    keep = np.linalg.norm(Xs[:, [3, 4]] - sample[None, [3, 4]], axis=-1).argsort()[k:]
    return Xs[keep], ys[keep]


def train_on_partial_data(path):
    (Xs, ys), (Xs_val, ys_val), (Xs_test, ys_test) = load_data(path, splitted=True)

    # Randomly pick a data point in Xs
    selected_sample = Xs[np.random.randint(low=0, high=Xs.shape[0])]
    for repetition in range(5):
        ks = [1_000, 2_500, 5_000, 7_500, 10_000, 15_000, 20_000, 25_000, 50_000, 100_000, 125_000, 150_000]
        for k in ks:
            # - Remove k nearest neighbors
            Xs, ys = remove_k_nearest_neighbors(Xs, ys, selected_sample, k)

            # - Train new VAE
            print(f"Repetition {repetition}: Now VAE with {k} samples less")
            vae_path = f"./rep_{repetition}_vae_{k}_nn_removed"
            if not os.path.isdir(vae_path):
                train_vae(Xs, Xs_val, Xs_test, logging_dir=vae_path)
                np.save(f"{vae_path}/Xs_train.npy", Xs)
                np.save(f"{vae_path}/ys_train.npy", ys)

            # - Train new surrogate model
            print(f"Repetition {repetition}: Now surrogate with {k} samples less")
            surrogate_path = f"rep_{repetition}_surrogate_{k}_nn_removed"
            if not os.path.isdir(surrogate_path):
                train_surrogate(
                    Xs, ys, Xs_val, ys_val, Xs_test, ys_test, dimensions=[32, 32], logging_dir=surrogate_path
                )

            # - Check quality of counterfactuals
            cf_deviations_file = f"./{vae_path}/rep_{repetition}_cf_performances_{k}_removed.npy"
            if not os.path.isfile(cf_deviations_file):
                abs_deviations = evaluate_cf_quality(Xs, ys, Xs_test, ys_test, vae_path, surrogate_path)
                np.save(cf_deviations_file, abs_deviations)


if __name__ == "__main__":
    root = "/home/andreas/Uni/projects/dfg_bending_simulation/data-generator-demo"

    # Idea is to leave out W1
    train_on_partial_data(path=f"{root}/data/R_20240719_results_uncertain_only_bo1.csv")
