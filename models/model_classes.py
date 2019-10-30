import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Input, Reshape, Dropout,
    BatchNormalization, Lambda, Dense, GRU)

tfd = tfp.distributions


def fvm_entropy(kappa):
    """For d=3"""
    expk2 = K.exp(- 2 * kappa)
    return (
        1 + np.log(2 * np.pi)
        - 2 * kappa * expk2 / (1 - expk2)
        + tf.math.log1p(- expk2)
        - K.log(kappa)
    )


def mean_neg_log_prob(y_true, dist_pred):
    return -K.mean(dist_pred.log_prob(y_true))



def mean_neg_dot_prod(y_true, y_pred):
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.mean(K.sum(y_true * y_pred, axis=1))


def neg_log_prob(y_true, dist_pred):
    return -dist_pred.log_prob(y_true)


def neg_dot_prod(y_true, y_pred):
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.sum(y_true * y_pred, axis=1)


class FisherVonMises(tfd.VonMisesFisher):
    """Numerically stable implementation for d=3"""

    def _entropy(self):
        return fvm_entropy(self.concentration)

    def _mean(self):
        kappa = self.concentration
        expk2 = K.exp(- 2 * kappa)
        W = (kappa * (1 + expk2) - (1 - expk2)) / (kappa * (1 - expk2))
        return W[..., tf.newaxis] * self.mean_direction

    def _log_normalization(self):
        kappa = self.concentration
        expk2 = K.exp(- 2 * kappa)
        return np.log(2*np.pi) + kappa + tf.math.log1p(- expk2) - K.log(kappa)
        

class FvM(object):
    """docstring for FvM"""
    model_name="FvM"

    custom_objects = {
            "mean_neg_log_prob": mean_neg_log_prob,
            "mean_neg_dot_prod": mean_neg_dot_prod,
            "DistributionLambda": tfp.layers.DistributionLambda
        }

    sample_class = "FvMSamples"

    summaries = "FvMSummaries"

    def __init__(self, input_shape, loss_weight=None, **kwargs):

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
        kappa = Lambda(lambda t: K.squeeze(t, 1) + 0.001, name="kappa")(kappa)

        return tfp.layers.DistributionLambda(
            make_distribution_fn=lambda params: FisherVonMises(
                mean_direction=params[0], concentration=params[1]),
            convert_to_tensor_fn=tfd.Distribution.mean,
            name="fvm"
        )([mu, kappa])

    def compile(self, optimizer):
        self.keras.compile(
            optimizer=optimizer,
            loss=self.custom_objects["mean_neg_log_prob"],
            metrics=[self.custom_objects["mean_neg_dot_prod"]])


class FvMHybrid(object):
    """docstring for FvMHybrid"""
    model_name="FvMHybrid"

    custom_objects = {
            "neg_log_prob": neg_log_prob,
            "DistributionLambda": tfp.layers.DistributionLambda
        }

    sample_class = "FvMHybridSamples"
    
    summaries = "FvMHybridSummaries"

    def __init__(self, input_shape, loss_weight, **kwargs):

        inputs = Input(shape=input_shape, name="inputs")
        shared = self._shared_layers(inputs)

        self.keras = tf.keras.Model(
            inputs,
            [self.fvm(shared), self.isterminal(shared)],
            name=self.model_name
        )

        self.loss_weight = loss_weight

    @staticmethod
    def _shared_layers(inputs):
        x = Dense(2048, activation="relu")(inputs)
        x = Dense(2048, activation="relu")(x)
        x = Dense(2048, activation="relu")(x)
        return x

    @staticmethod
    def fvm(x):
        mu = Dense(1024, activation="relu")(x)
        mu = Dense(3, activation="linear")(mu)
        mu = Lambda(lambda t: K.l2_normalize(t, axis=-1), name="mu")(mu)

        kappa = Dense(1024, activation="relu")(x)
        kappa = Dense(1, activation="relu")(kappa)
        kappa = Lambda(lambda t: K.squeeze(t, 1) + 0.001, name="kappa")(kappa)

        fvm = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda params: FisherVonMises(
                mean_direction=params[0], concentration=params[1]),
            convert_to_tensor_fn=tfd.Distribution.mean,
            name="fvm"
        )([mu, kappa])

        return fvm

    @staticmethod
    def isterminal(x):
        x = Dense(1024, activation="relu")(x)
        x = Dense(1, activation="sigmoid", name="isterminal")(x)
        return x

    def compile(self, optimizer):
        self.keras.compile(
            optimizer=optimizer,
            loss={
                "fvm": self.custom_objects["neg_log_prob"],
                "isterminal": "binary_crossentropy"
            },
            loss_weights = {"fvm": 1.0, "isterminal": self.loss_weight},
        )


class RNNModel(object):
    model_name="RNNModel"

    sample_class = "RNNSamples"

    def __init__(self, input_shape, batch_size, **kwargs):
        inputs = Input(shape=input_shape, batch_size=batch_size, name="inputs")
        self.keras = tf.keras.Model(inputs, self.model_fn(inputs), name=self.model_name)

    @staticmethod
    def model_fn(inputs):
        hidden_size = [500, 500]  # Fixed

        x = GRU(hidden_size[0], return_sequences=True, stateful=True)(inputs)
        if len(hidden_size) > 1:
            for hidden_size in hidden_size[1:-1]:
                x = GRU(hidden_size, return_sequences=True, stateful=True)(x)
            x = GRU(hidden_size[-1], return_sequences=True, stateful=True)(x)
        x = Dense(3, activation='linear', name='output1')(x)  # TODO: This output is not fed to model, make sure it's fine
        return x

    def compile(self, optimizer):
        self.keras.compile(
            optimizer=optimizer,
            loss = {'output1': 'mean_squared_error'})


def fvm_cost(y_true, dist_pred):
    return - K.sum(dist_pred.mean() * y_true, axis=1)


class FvMEntropyRegularizer(tf.keras.regularizers.Regularizer):
    """docstring for FvMEntropyRegularizer"""
    def __init__(self, temperature):
        super(FvMEntropyRegularizer, self).__init__()
        self.temperature = K.cast_to_floatx(temperature)
        
    def __call__(self, kappa):
        return - self.temperature * K.mean(fvm_entropy(kappa))

    def get_config(self):
        return {"temperature": float(self.temperature)}

    def set_T(self, T):
        self.temperature = K.cast_to_floatx(T)


class FvMEntropyRegularization(tf.keras.layers.Layer):
    """"""
    def __init__(self, temperature, **kwargs):
        super(FvMEntropyRegularization, self).__init__(
            activity_regularizer=FvMEntropyRegularizer(temperature), **kwargs)
        self.supports_masking = True
        self.temperature = temperature

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"temperature": self.temperature}
        base_config = super(FvMEntropyRegularization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Entrack(FvM):
    """docstring for Entrack"""
    model_name="Entrack"

    custom_objects = {
            "mean_free_energy": None, # set during by init
            "mean_neg_dot_prod": mean_neg_dot_prod,
            "DistributionLambda": tfp.layers.DistributionLambda
        }
  
    @staticmethod
    def mean_free_energy(T):
        loss_fn = lambda y_true, dist_pred: K.mean(
            - K.sum(dist_pred.mean() * y_true, axis=1) - T * dist_pred.entropy()
        )
        return loss_fn

    def __init__(self, *args, temperature, **kwargs):

        super(Entrack, self).__init__(*args, **kwargs)

        self.custom_objects["mean_free_energy"] = self.mean_free_energy(
            temperature)

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
        kappa = Lambda(lambda t: K.squeeze(t, 1) + 0.001, name="kappa")(kappa)

        fvm = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda params: FisherVonMises(
                mean_direction=params[0], concentration=params[1]),
            convert_to_tensor_fn=tfd.Distribution.mean,
            name="fvm"
        )([mu, kappa])

        return fvm

    def compile(self, optimizer):
        self.keras.compile(
            optimizer=optimizer,
            loss=self.custom_objects["mean_free_energy"],
            metrics=[self.custom_objects["mean_neg_dot_prod"]])
