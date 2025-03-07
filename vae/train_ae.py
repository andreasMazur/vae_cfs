from data.data_loading import COL_BENDING_ANGLE, COL_BENDING_RADIUS, USE_CLAMPING_FILTER, COL_CLAMPING
from vae.variational_autoencoder import (
    VariationalAutoEncoder,
    EvidenceLowerBound,
    ReconstructionMetric,
    KLDivMetric
)

import tensorflow as tf
import os


def normalize_data(data, disregard_dims=None):
    dims_mean = data.mean(axis=0)
    dims_std = data.std(axis=0)
    if disregard_dims is not None:
        dims_mean[disregard_dims] = 0.
        dims_std[disregard_dims] = 1.
    data = (data - dims_mean) / dims_std
    return data, dims_mean, dims_std


def de_normalize_data(data, dims_mean, dims_std):
    return data * dims_std + dims_mean


def train_vae(Xs, Xs_val, Xs_test, logging_dir=None):
    if logging_dir is None:
        logging_dir = "./trained_vae"

    # Don't normalize clamping
    if USE_CLAMPING_FILTER is None:
        Xs, Xs_means, Xs_stds = normalize_data(Xs, disregard_dims=[COL_CLAMPING])
        Xs_val, Xs_means, Xs_stds = normalize_data(Xs_val, disregard_dims=[COL_CLAMPING])
        Xs_test, Xs_means, Xs_stds = normalize_data(Xs_test, disregard_dims=[COL_CLAMPING])
    else:
        Xs, Xs_means, Xs_stds = normalize_data(Xs)
        Xs_val, Xs_means, Xs_stds = normalize_data(Xs_val)
        Xs_test, Xs_means, Xs_stds = normalize_data(Xs_test)

    latent_dim = 2
    vae = VariationalAutoEncoder(encoding_dims=[16, 16, latent_dim], decoding_dims=[16, 16])
    vae.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.01532879826662228,
                decay_steps=2074,
                decay_rate=0.9038763890610878
            ),
            weight_decay=0.4152095947945318
        ),
        loss=EvidenceLowerBound(latent_dim=latent_dim, beta=92.81073868431973, warmup_steps=1000),
        metrics=[ReconstructionMetric(latent_dim=latent_dim), KLDivMetric(latent_dim=latent_dim)]
    )
    vae(tf.zeros((1, Xs.shape[-1])))
    # vae.summary()

    # Start training
    os.makedirs(logging_dir, exist_ok=True)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{logging_dir}/model_ckp.keras",
        monitor="val_reconstruction_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=False
    )
    stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_reconstruction_loss",
        mode="min",
        patience=10,
        min_delta=0.001
    )
    csv_callback = tf.keras.callbacks.CSVLogger(f"{logging_dir}/training.csv")
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=f"{logging_dir}/tensorboard_logs",
        histogram_freq=0,
        write_graph=False,
        write_images=False,
        write_steps_per_second=False,
        update_freq="batch"
    )
    training_history = vae.fit(
        Xs,
        Xs,
        batch_size=64,
        epochs=10_000,
        validation_data=(Xs_val, Xs_val),
        callbacks=[cp_callback, csv_callback, stop_callback, tensorboard_cb],
        verbose=0
    )
    return training_history
