from compute_cf.combined_model import CombinedModel
from vae.train_ae import normalize_data

import numpy as np
import tensorflow as tf
import sys


def compute_counterfactual(Xs_train,
                           ys_train,
                           vae_path,
                           surrogate_path,
                           target_value,
                           allowed_deviation=0.1,
                           allowed_init_deviation=1.0,
                           eta=0.01,
                           max_iterations=2000,
                           verbose=False,
                           restart_if_necessary=False):
    """Compute a counterfactual process configuration for a given target outcome bending angle.

    Parameters
    ----------
    Xs_train: np.ndarray
        The training data features (process configurations).
    ys_train: np.ndarray
        The training data targets (outcome bending angles).
    vae_path: str
        The path to the trained VAE model.
    surrogate_path: str
        The path to the trained surrogate model.
    target_value: float
        The target outcome bending angle for which a counterfactual process configuration is to be computed.
    allowed_deviation: float
        The maximum allowed deviation between the predicted outcome bending angle of the counterfactual and the target
        value.
    allowed_init_deviation: float
        The maximum allowed deviation between the initial predicted outcome bending angle and the target value. If this
        is exceeded after a preset amount of iterations, the computation is restarted if 'restart_if_necessary' is set
        to 'True'.
    eta: float
        The step size for the gradient descent optimization.
    max_iterations: int
        The maximum number of iterations for the gradient descent optimization.
    verbose: bool
        Whether to print information during the computation of the counterfactual.
    restart_if_necessary: bool
        Whether to restart the computation if the initial deviation is too high.

    Returns
    -------
    (np.ndarray, float, bool):
        A tuple containing the counterfactual process configuration, the predicted outcome bending angle of the
        counterfactual and a flag indicating whether the computation needs to be restarted.
    """

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

    # Generate counterfactual (gradient descent in latent space)
    deviation = np.inf
    step = 0
    denormalized_config, denormalized_regr = None, None
    while deviation > allowed_deviation and step < max_iterations:
        with tf.GradientTape() as tape:
            tape.watch(init_mean)
            conf, regression = combined_model(init_mean)
            regression = regression[0]
            squared_diff = tf.math.squared_difference(regression, normalized_target_value)

        grad = tape.gradient(squared_diff, init_mean)
        init_mean = init_mean - eta * grad

        denormalized_regr = regression * ys_stds[0] + ys_means[0]
        deviation = np.abs(denormalized_regr - target_value)
        # Stop computation of counterfactual if sample has been sampled too far away from suitable configuration
        if restart_if_necessary and step == max_iterations / 4 and deviation > allowed_init_deviation:
            # Indicate that run shall be restarted by return 'True'
            return None, None, True

        denormalized_config = conf[0] * Xs_stds + Xs_means

        if verbose:
            sys.stdout.write(
                f"\rStep: {step} - "
                f"Generated config: {denormalized_config.numpy().tolist()} - "
                f"Regression: {denormalized_regr.numpy()[0]:.3f} - "
                f"Target value: {target_value[0]:.3f} - "
                f"Deviation: {deviation[0]:.3f}"
            )
        step += 1

    # Indicate that does not need to be restarted by return 'False'
    return denormalized_config, denormalized_regr, False


def compute_cf_wrapper(Xs, ys, Xs_test, ys_test, vae_path, surrogate_path, N_test=200, max_cf_trials=5, verbose=True):
    """A wrapper function to compute counterfactuals for multiple target outcome bending angles.

    Parameters
    ----------
    Xs: np.ndarray
        The training data features (process configurations).
    ys: np.ndarray
        The training data targets (outcome bending angles).
    Xs_test: np.ndarray
        The test data features (process configurations) which are compared against the counterfactuals and whose outcome
        bending angles are taken as target values.
    ys_test: np.ndarray
        The test data targets (outcome bending angles) which are used as target values for the counterfactuals.
    vae_path: str
        The path to the trained VAE model.
    surrogate_path: str
        The path to the trained surrogate model.
    N_test: int
        The number of test samples to compute counterfactuals for.
    max_cf_trials: int
        The maximum number of trials to compute a counterfactual for a specific target value.
    verbose: bool
        Whether to print information during computation of counterfactual.

    Returns
    -------
    (np.ndarray, np.ndarray):
        A tuple containing an array of counterfactual process configurations and the array of the predicted outcomes for
        the counterfactuals.
    """
    # Normalize data using z-score normalization
    _, Xs_means, Xs_stds = normalize_data(Xs)

    # Filter down test sample to 'N_test' samples
    Xs_test, ys_test = Xs_test[:N_test], ys_test[:N_test]

    # Try to re-create original W1-values and compare CF to original data point
    cfs, cf_preds = [], []
    for idx, (target_cf, target_value) in enumerate(zip(Xs_test, ys_test)):
        trial = 0
        restart_cf_run = True
        while restart_cf_run and trial < max_cf_trials:
            if verbose:
                print(f"\nTrial: {trial}")

            # Compute counterfactual explanation for target value
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
