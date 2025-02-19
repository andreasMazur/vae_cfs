import tensorflow as tf


class VariationalAutoEncoder(tf.keras.models.Model):
    def __init__(self, encoding_dims, decoding_dims, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Encoder
        self.encoding_dims = encoding_dims
        self.encoder = tf.keras.models.Sequential(name="encoder")
        for dim in self.encoding_dims[:-1]:
            self.encoder.add(tf.keras.layers.Dense(dim, activation="linear"))
            self.encoder.add(tf.keras.layers.BatchNormalization())
            self.encoder.add(tf.keras.layers.ReLU())

        # Bottleneck
        self.latent_dim = self.encoding_dims[-1]
        self.mean_predictor = tf.keras.layers.Dense(
            self.latent_dim,
            activation="linear",
            name="mean_predictor"
        )
        self.log_var_predictor = tf.keras.layers.Dense(
            self.latent_dim,
            activation="linear",
            name="log_std_predictor"
        )

        # Decoder
        self.decoding_dims = decoding_dims
        self.decoder = tf.keras.models.Sequential(name="decoder")
        for dim in self.decoding_dims:
            self.decoder.add(tf.keras.layers.Dense(dim, activation="linear"))
            self.decoder.add(tf.keras.layers.BatchNormalization())
            self.decoder.add(tf.keras.layers.ReLU())

        self.tool_geometry_output = tf.keras.layers.Dense(2, activation="linear")
        self.clamping_output = tf.keras.layers.Dense(1, activation="linear")
        self.feature_amount = None

    def build(self, input_shape):
        self.feature_amount = input_shape[1]
        self.call(tf.zeros(input_shape))

    def encode(self, inputs):
        inputs = self.encoder(inputs)
        pred_mean = self.mean_predictor(inputs)
        pred_log_var = self.log_var_predictor(inputs)
        return pred_mean, pred_log_var

    def sample(self, pred_mean, pred_log_var, training=None):
        if training:
            epsilon = tf.random.normal(shape=tf.shape(pred_mean), mean=0.0, stddev=1.0)
        else:
            epsilon = 0.
        return pred_mean + tf.math.exp(pred_log_var / 2) * epsilon

    def decode(self, pred_mean, pred_log_var, training=None):
        samples = self.sample(pred_mean, pred_log_var, training=training)
        samples = self.decoder(samples)
        reconstructed_tool_geometry = self.tool_geometry_output(samples)
        reconstructed_clamping = self.clamping_output(samples)
        return tf.concat([reconstructed_clamping, reconstructed_tool_geometry], axis=-1)

    def call(self, inputs, training=None, mask=None):
        pred_mean, pred_log_var = self.encode(inputs)
        reconstruction = self.decode(pred_mean, pred_log_var, training)
        return tf.concat([pred_mean, pred_log_var, reconstruction], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "encoding_dims": self.encoding_dims,
                "decoding_dims": self.decoding_dims,
                "feature_amount": self.feature_amount
            }
        )
        return config

    @classmethod
    def from_config(cls, config, **kwargs):
        model = cls(encoding_dims=config["encoding_dims"], decoding_dims=config["decoding_dims"], **kwargs)
        # For 'tf.keras.models.load_model' to assign the trained weights properly one needs to initialize
        # the weights by calling the model once with dummy data.
        model.build([1, config["feature_amount"]])
        return model


class ReconstructionMetric(tf.keras.metrics.Metric):
    def __init__(self, latent_dim, name="reconstruction_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_loss = self.add_weight(name="total_loss", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
        self.latent_dim = latent_dim

    def update_state(self, y_true, y_pred, *args, **kwargs):
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        clamping = y_pred[:, self.latent_dim * 2]
        tool_geometry = y_pred[:, self.latent_dim * 2 + 1:]

        # Sum over batch / total loss over batch
        bce = binary_cross_entropy(y_true[:, 0], clamping)
        mse = mean_squared_error(y_true[:, 1:], tool_geometry)
        total_recon_loss = tf.reduce_sum((bce / 3) + (2 * mse / 3))
        self.total_loss.assign_add(total_recon_loss)
        self.count.assign_add(batch_size)

    def result(self):
        return self.total_loss / self.count

    def reset_state(self):
        self.total_loss.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"latent_dim": self.latent_dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class KLDivMetric(tf.keras.metrics.Metric):
    def __init__(self, latent_dim, name="kl_divergence", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_loss = self.add_weight(name="total_loss", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
        self.latent_dim = latent_dim

    def update_state(self, y_true, y_pred, *args, **kwargs):
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        pred_mean = y_pred[:, :self.latent_dim]
        pred_var = y_pred[:, self.latent_dim:self.latent_dim * 2]

        # Sum over batch / total loss over batch
        kl_div = tf.reduce_sum(analytical_kl_div(pred_mean, pred_var))
        self.total_loss.assign_add(kl_div)
        self.count.assign_add(batch_size)

    def result(self):
        return self.total_loss / self.count

    def reset_state(self):
        self.total_loss.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"latent_dim": self.latent_dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class EvidenceLowerBound(tf.keras.losses.Loss):
    def __init__(self, latent_dim, beta, warmup_steps, name="evidence_lower_bound_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.beta = beta
        self.training_steps = 0
        self.warmup_steps = warmup_steps

    def call(self, y_true, y_pred):
        pred_mean = y_pred[:, :self.latent_dim]
        pred_var = y_pred[:, self.latent_dim:self.latent_dim * 2]
        clamping = y_pred[:, self.latent_dim * 2]
        tool_geometry = y_pred[:, self.latent_dim * 2 + 1:]

        # KL-divergence regularization + reconstruction error (bce + mse)
        kl_div = analytical_kl_div(pred_mean, pred_var)
        bce = binary_cross_entropy(y_true[:, 0], clamping)
        mse = mean_squared_error(y_true[:, 1:], tool_geometry)

        # Respect warm-up period
        coeff = self.training_steps / self.warmup_steps if self.training_steps < self.warmup_steps else 1.0
        self.training_steps = self.training_steps + 1

        loss = coeff * self.beta * kl_div + (bce / 3) + (2 * mse / 3)
        return loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "beta": self.beta,
            "warmup_steps": self.warmup_steps,
            "training_steps": self.training_steps
        })
        return config

    @classmethod
    def from_config(cls, config):
        elbo = cls(
            latent_dim=config["latent_dim"],
            beta=config["beta"],
            warmup_steps=config["warmup_steps"],
        )
        elbo.training_steps = config["training_steps"]
        return elbo


def analytical_kl_div(pred_mean, pred_var):
    return -(1 / 2) * tf.reduce_sum(1 + pred_var - pred_mean ** 2 - tf.math.exp(pred_var), axis=-1)


def mean_squared_error(y_true, reconstruction):
    # First average over epsilon-samples, then compute MSE for instances in batch
    return tf.reduce_mean(tf.math.squared_difference(y_true, reconstruction), axis=-1)


def binary_cross_entropy(y_true, reconstruction):
    # Assumes log-output for reconstruction
    probabilities = tf.clip_by_value(tf.math.sigmoid(reconstruction), clip_value_min=1e-7, clip_value_max=1 - 1e-7)
    return -(y_true * tf.math.log(probabilities) + (1 - y_true) * tf.math.log(1 - probabilities))
