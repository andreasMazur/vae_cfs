from data.data_loading import load_data
from vae.train_ae import de_normalize_data, normalize_data
from vae.variational_autoencoder import (
    VariationalAutoEncoder,
    ReconstructionMetric,
    KLDivMetric,
    EvidenceLowerBound
)
from surrogate_model.surrogat_model import r2_score

import tensorflow as tf


class CombinedModel(tf.keras.models.Model):
    def __init__(self, init_data, vae_path, surrogate_path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize variational autoencoder
        self.vae = tf.keras.models.load_model(
            f"{vae_path}/model_ckp.keras",
            custom_objects={
                "VariationalAutoEncoder": VariationalAutoEncoder,
                "ReconstructionMetric": ReconstructionMetric,
                "KLDivMetric": KLDivMetric,
                "EvidenceLowerBound": EvidenceLowerBound
            }
        )
        pred_mean, _ = self.vae.encode(init_data)
        self.minimal_mean = tf.reduce_min(pred_mean, axis=0)
        self.maximum_mean = tf.reduce_max(pred_mean, axis=0)

        # Initialize surrogate model
        self.surrogate = tf.keras.models.load_model(
            f"{surrogate_path}/surrogate.keras", custom_objects={"r2_score": r2_score}
        )

    def call(self, inputs, **kwargs):
        mean_values, material_information = inputs
        generated = self.vae.decode(pred_mean=mean_values, pred_log_var=tf.zeros_like(mean_values))
        input_for_surrogate = tf.concat(
            [material_information[None, :], tf.round(tf.nn.sigmoid(generated[:, :1])), generated[:, 1:]], axis=-1
        )
        return generated, self.surrogate(input_for_surrogate)

    def return_in_space_mean(self, n=1):
        return tf.random.uniform((n, self.vae.latent_dim), minval=self.minimal_mean, maxval=self.maximum_mean)


def test_cm(data_path, vae_path, surrogate_path, repetitions=10):
    (Xs, ys), _, _ = load_data(data_path, splitted=True)

    Xs, Xs_means, Xs_stds = normalize_data(Xs)
    ys, ys_means, ys_stds = normalize_data(ys)
    cm = CombinedModel(Xs, vae_path, surrogate_path)
    cm.return_in_space_mean()

    sampled_means = tf.random.uniform((repetitions, tf.shape(cm.minimal_mean)[0]), cm.minimal_mean, cm.maximum_mean)
    artificial_configs, regressions = cm(sampled_means)
    artificial_configs = de_normalize_data(artificial_configs, Xs_means, Xs_stds)
    regressions = de_normalize_data(regressions, ys_means, ys_stds)

    for i in range(repetitions):
        print(artificial_configs[i].numpy().tolist())
        print(regressions[i].numpy().tolist())
        print()
