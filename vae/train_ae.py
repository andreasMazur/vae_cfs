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
        dims_mean[disregard_dims] = 0
        dims_std[disregard_dims] = 1
    data = (data - dims_mean) / dims_std
    return data, dims_mean, dims_std


def de_normalize_data(data, dims_mean, dims_std):
    return data * dims_std + dims_mean


def train_vae(Xs, Xs_val, Xs_test, logging_dir=None):
    if logging_dir is None:
        logging_dir = "./trained_vae"

    # Filter for clamping angle and radius
    Xs, Xs_val, Xs_test = Xs[:, [2, 3, 4]], Xs_val[:, [2, 3, 4]], Xs_test[:, [2, 3, 4]]

    # Don't normalize clamping
    Xs, Xs_means, Xs_stds = normalize_data(Xs, disregard_dims=[0])
    Xs_val, Xs_means, Xs_stds = normalize_data(Xs_val, disregard_dims=[0])
    Xs_test, Xs_means, Xs_stds = normalize_data(Xs_test, disregard_dims=[0])

    latent_dim = 2
    vae = VariationalAutoEncoder(encoding_dims=[16, 16, latent_dim], decoding_dims=[16, 16])
    vae.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.01431224851518938,
                decay_steps=2407,
                decay_rate=0.9508174254269613
            ),
            weight_decay=0.12546620417641197
        ),
        loss=EvidenceLowerBound(latent_dim=latent_dim, beta=55.29819837741578, warmup_steps=576),
        metrics=[ReconstructionMetric(latent_dim=latent_dim), KLDivMetric(latent_dim=latent_dim)]
    )
    vae(tf.zeros((1, Xs.shape[-1])))
    vae.summary()

    # Start training
    os.makedirs(logging_dir, exist_ok=True)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{logging_dir}/model_ckp.keras",
        monitor="val_reconstruction_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=True
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
    vae.fit(
        Xs,
        Xs,
        batch_size=64,
        epochs=10_000,
        validation_data=(Xs_val, Xs_val),
        callbacks=[cp_callback, csv_callback, stop_callback, tensorboard_cb]
    )

    print("Evaluate")
    result = vae.evaluate(x=Xs_test, y=Xs_test)
    print(dict(zip(vae.metrics_names, result)))
