from data.data_loading import USE_CLAMPING_FILTER
from vae.variational_autoencoder import (
    VariationalAutoEncoder,
    ReconstructionMetric,
    KLDivMetric,
    EvidenceLowerBound
)
from surrogate_model.surrogate_model import r2_score

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

    def call(self, inputs, training=None, **kwargs):
        generated = self.vae.decode(pred_mean=inputs, pred_log_var=tf.zeros_like(inputs), training=training)
        if USE_CLAMPING_FILTER is None:
            generated = tf.concat(
                [generated[:, :2], tf.round(tf.nn.sigmoid(generated[:, 2:3])), generated[:, 3:]], axis=-1
            )
        return generated, self.surrogate(generated)

    def return_in_space_mean(self, n=1):
        return tf.random.uniform((n, self.vae.latent_dim), minval=self.minimal_mean, maxval=self.maximum_mean)
