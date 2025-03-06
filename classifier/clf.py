from data.data_loading import USE_CLAMPING_FILTER, COL_CLAMPING
from vae.train_ae import normalize_data

import tensorflow as tf
import keras
import os


def train_clf(Xs, Xs_removed, logging_dir, dimensions=None):
    if dimensions is None:
        dimensions = [64 for _ in range(5)]

    # Set binary labels - 0: removed | 1: not removed / available
    ys = tf.ones((Xs.shape[0], 1))
    ys_removed = tf.zeros((Xs_removed.shape[0], 1))
    concatenated = tf.concat([tf.concat([Xs, ys], axis=-1), tf.concat([Xs_removed, ys_removed], axis=-1)], axis=0)
    concatenated = tf.random.shuffle(concatenated)
    Xs, ys = concatenated[:, :-1], concatenated[:, None, -1]

    # Determine split indices
    train_split = int(Xs.shape[0] * 0.7)
    val_split = train_split + int(Xs.shape[0] * 0.15)

    # Set splits
    Xs_train, ys_train = Xs[:train_split].numpy(), ys[:train_split].numpy()
    Xs_val, ys_val = Xs[train_split:val_split].numpy(), ys[train_split:val_split].numpy()
    Xs_test, ys_test = Xs[val_split:].numpy(), ys[val_split:].numpy()

    # Normalize data
    if USE_CLAMPING_FILTER is None:
        Xs_train, Xs_train_mean, Xs_train_stds = normalize_data(Xs_train, disregard_dims=[COL_CLAMPING])
        Xs_val, Xs_val_mean, Xs_val_stds = normalize_data(Xs_val, disregard_dims=[COL_CLAMPING])
        Xs_test, Xs_test_mean, Xs_test_stds = normalize_data(Xs_test, disregard_dims=[COL_CLAMPING])
    else:
        Xs_train, Xs_train_mean, Xs_train_stds = normalize_data(Xs_train)
        Xs_val, Xs_val_mean, Xs_val_stds = normalize_data(Xs_val)
        Xs_test, Xs_test_mean, Xs_test_stds = normalize_data(Xs_test)

    # Define classifier
    layers = [keras.layers.Dense(dim, activation="relu") for dim in dimensions]
    clf = keras.models.Sequential(layers + [keras.layers.Dense(1, activation="sigmoid")])
    clf.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )

    # Train regressor
    os.makedirs(logging_dir, exist_ok=True)
    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath=f"{logging_dir}/clf.keras",
        save_best_only=True,
        save_weights_only=False,
        verbose=True
    )
    csv_callback = keras.callbacks.CSVLogger(f"{logging_dir}/training.csv")
    stop_callback = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, min_delta=0.01)
    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir=f"{logging_dir}/tensorboard_logs",
        histogram_freq=0,
        write_graph=False,
        write_images=False,
        write_steps_per_second=False,
        update_freq="batch"
    )
    clf.fit(
        Xs_train,
        ys_train,
        batch_size=64,
        epochs=10_000,
        validation_data=(Xs_val, ys_val),
        callbacks=[cp_callback, csv_callback, stop_callback, tensorboard_cb]
    )
    clf.save(f"{logging_dir}/trained_clf.keras")
    clf.evaluate(Xs_test, ys_test)
