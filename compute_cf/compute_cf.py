from compute_cf.combined_model import CombinedModel
from vae.train_ae import normalize_data

import numpy as np
import tensorflow as tf
import sys


def compute_counterfactual(Xs_train,
                           ys_train,
                           vae_path,
                           surrogate_path,
                           material_information,
                           target_value,
                           allowed_deviation=0.1,
                           eta=0.01,
                           max_iterations=2000,
                           verbose=False):
    # Load and normalize data
    Xs_train, Xs_means, Xs_stds = normalize_data(Xs_train)
    ys_train, ys_means, ys_stds = normalize_data(ys_train)

    # Get model
    combined_model = CombinedModel(Xs_train, vae_path, surrogate_path)
    combined_model.vae.trainable = False
    combined_model.surrogate.trainable = False

    # Random initial latent mean
    init_mean = combined_model.return_in_space_mean()

    # Normalize target value
    normalized_target_value = (target_value - ys_means[0]) / ys_stds[0]

    # Generate counterfactual
    deviation = np.inf
    step = 0
    denormalized_config, denormalized_regr = None, None
    while deviation > allowed_deviation and step < max_iterations:
        with tf.GradientTape() as tape:
            tape.watch(init_mean)

            conf, regression = combined_model([init_mean, material_information])
            conf = conf[0]
            regression = regression[0, 0]
            squared_diff = tf.math.squared_difference(regression, normalized_target_value)
            grad = tape.gradient(squared_diff, init_mean)

            init_mean = init_mean - eta * grad

        denormalized_regr = regression * ys_stds[0] + ys_means[0]
        deviation = np.abs(denormalized_regr - target_value)

        denormalized_config = conf * Xs_stds + Xs_means

        if verbose:
            sys.stdout.write(
                f"\rStep: {step} - "
                f"Generated config: {denormalized_config.numpy().tolist()} - "
                f"Regression: {denormalized_regr:.3f} - "
                f"Target value: {target_value[0]:.3f} - "
                f"Deviation: {deviation[0]:.3f}"
            )
        step += 1
    denormalized_config = tf.concat(
        [tf.round(tf.nn.sigmoid(denormalized_config[:1])), denormalized_config[1:]], axis=-1
    )
    return denormalized_config, denormalized_regr
