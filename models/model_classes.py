import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Input, Reshape, Dropout,
    BatchNormalization, Lambda, Dense)

tfd = tfp.distributions

class EntrackConditional(object):
    """docstring for EntrackConditional"""
    model_name="EntrackConditional"

    def __init__(self, input_shape):

        inputs = Input(shape=input_shape, name="inputs")

        self.keras = tf.keras.Model(
            inputs,
            self.model_fn(inputs),
            name=self.model_name
        )

    @staticmethod
    def model_fn(inputs):
        """MLP with two output heads for mu and kappa"""
        x = Dense(2048, activation="relu")(inputs)
        x = Dense(2048, activation="relu")(x)
        x = Dense(2048, activation="relu")(x)

        mu = Dense(1024, activation="relu")(x)
        mu = Dense(3, activation="linear")(mu)
        mu = Lambda(lambda t: K.l2_normalize(t, axis=-1), name="mu")(mu)

        kappa = Dense(1024, activation="relu")(x)
        kappa = Dense(1, activation="relu")(kappa)
        kappa = Lambda(lambda t: K.squeeze(t, 1), name="kappa")(kappa)

        return tfp.layers.DistributionLambda(
            make_distribution_fn=lambda params: tfd.VonMisesFisher(
                mean_direction=params[0], concentration=params[1]),
            convert_to_tensor_fn=tfd.Distribution.mean
        )([mu, kappa])

    @staticmethod
    def loss(observed_y, predicted_distribution):
        """Negative log-likelihood"""
        return -K.mean(predicted_distribution.log_prob(observed_y))

    @staticmethod
    def neg_dot_prod(y_true, y_pred):
        y_pred = K.l2_normalize(y_pred, axis=-1)
        return -K.mean(K.sum(y_true * y_pred, axis=1))

    def _metrics(self):
        return [self.neg_dot_prod]

    def compile(self, optimizer):
        self.keras.compile(
            optimizer=optimizer,
            loss=self.loss,
            metrics=self._metrics())

    def custom_objects(self):
        return {"loss": self.loss,
                "DistributionLambda": tfp.layers.DistributionLambda}