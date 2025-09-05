from vae.variational_autoencoder import (
    VariationalAutoEncoder,
    ReconstructionMetric,
    KLDivMetric,
    EvidenceLowerBound
)
from surrogate_model.surrogate_model import r2_score

import tensorflow as tf


class CombinedModel(tf.keras.models.Model):
    """A combined model consisting of a pre-trained VAE and a pre-trained surrogate model.

    Attributes
    ----------
    vae: VariationalAutoEncoder
        The pre-trained variational autoencoder.
    surrogate: tf.keras.Model
        The pre-trained surrogate model.
    minimal_mean: tf.Tensor
        The minimum latent mean observed in the training data per latent dimension.
    maximum_mean: tf.Tensor
        The maximum latent mean observed in the training data per latent dimension.
    """

    def __init__(self, init_data, vae_path, surrogate_path, *args, **kwargs):
        """Initialize the CombinedModel with pre-trained VAE and surrogate model.

        Parameters
        ----------
        init_data: np.ndarray
            The training used to train the VAE and the surrogate model. This is used to determine the latent space
            boundaries within which initial counterfactuals are sampled.
        vae_path: str
            The path to the where the VAE model is stored.
        surrogate_path: str
            The path to the where the surrogate model is stored.
        """
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

    def call(self, inputs, training=None):
        """Forward pass through the VAE and the surrogate model.

        Parameters
        ----------
        inputs: tf.Tensor
            The process configurations for which the outcomes are to be predicted.
        training: bool | None
            The training mode flag.
        """
        generated = self.vae.decode(pred_mean=inputs, pred_log_var=tf.zeros_like(inputs), training=training)
        return generated, self.surrogate(generated)

    def return_in_space_mean(self, n=1):
        """Sample 'n' points uniformly in the latent space bounded by the minimum and maximum latent means.

        Parameters
        ----------
        n: int
            The number of points to sample.
        """
        return tf.random.uniform((n, self.vae.latent_dim), minval=self.minimal_mean, maxval=self.maximum_mean)
