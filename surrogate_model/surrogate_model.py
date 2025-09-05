from vae.train_ae import normalize_data, de_normalize_data

import tensorflow as tf
import keras
import numpy as np
import os


def evaluate(model, test_data):
    Xs_test, ys_test = test_data
    Xs_test, Xs_test_means, Xs_test_stds = normalize_data(Xs_test)
    ys_test, ys_test_means, ys_test_stds = normalize_data(ys_test)

    model.evaluate(x=Xs_test, y=ys_test)

    predictions = model(Xs_test).numpy()
    predictions = de_normalize_data(predictions, ys_test_means, ys_test_stds)
    ys_test = de_normalize_data(ys_test, ys_test_means, ys_test_stds)
    error = np.mean(np.abs(predictions - ys_test), axis=0)
    print(error)


def r2_score(y_true, y_pred):
    """Calculate the R^2 (coefficient of determination) regression score function.

    Parameters
    ----------
    y_true: np.ndarray
        The ground truth target values.
    y_pred: np.ndarray
        The predicted target values.

    Returns
    -------
    float:
        The R^2 score.
    """
    residual_sum_of_squares = tf.reduce_sum(tf.square(y_true - y_pred))
    total_sum_of_squares = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    r2 = 1 - residual_sum_of_squares / (total_sum_of_squares + keras.backend.epsilon())
    return r2


def train_surrogate(Xs, ys, Xs_val, ys_val, mlp_layer_dims=None, logging_dir=None):
    """Train a surrogate model to predict target angles from process configurations.

    Parameters
    ----------
    Xs: np.ndarray
        The training data features (process configurations).
    ys: np.ndarray
        The target data (outcome bending angles).
    Xs_val: np.ndarray
        The validation data features (process configurations).
    ys_val: np.ndarray
        The validation target data (outcome bending angles).
    mlp_layer_dims: list
        A list of integers defining the number of neurons in each hidden layer of the MLP.
    logging_dir: str
        The directory to save the trained model and training logs.
    """
    if mlp_layer_dims is None:
        mlp_layer_dims = [32, 32]
    if logging_dir is None:
        logging_dir = "./trained_surrogate"

    # Normalize data using z-score normalization
    Xs, Xs_mins, Xs_maxs = normalize_data(Xs)
    Xs_val, Xs_val_mins, Xs_val_maxs = normalize_data(Xs_val)
    ys, ys_means, ys_stds = normalize_data(ys)
    ys_val, ys_val_mins, ys_val_maxs = normalize_data(ys_val)

    # Define regressor
    layers = [keras.layers.Dense(dim, activation="relu") for dim in mlp_layer_dims]
    regressor = keras.models.Sequential(layers + [keras.layers.Dense(1, activation="linear")])
    regressor.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss="mse",
        metrics=["mse", "mae", r2_score]
    )

    # Train regressor
    os.makedirs(logging_dir, exist_ok=True)
    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath=f"{logging_dir}/surrogate.keras",
        save_best_only=True,
        save_weights_only=False,
        verbose=False
    )
    csv_callback = keras.callbacks.CSVLogger(f"{logging_dir}/training.csv")
    stop_callback = keras.callbacks.EarlyStopping(monitor="val_mse", patience=4, min_delta=0.001)
    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir=f"{logging_dir}/tensorboard_logs",
        histogram_freq=0,
        write_graph=False,
        write_images=False,
        write_steps_per_second=False,
        update_freq="batch"
    )
    regressor.fit(
        Xs,
        ys,
        batch_size=64,
        epochs=10_000,
        validation_data=(Xs_val, ys_val),
        callbacks=[cp_callback, csv_callback, stop_callback, tensorboard_cb],
        verbose=0
    )
    regressor.save(f"{logging_dir}/trained_surrogate.keras")
