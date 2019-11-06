import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Input, Reshape, Dropout,
    BatchNormalization, Lambda, Dense, GRU)

from utils.config import deep_update
from utils.training import Temperature

from utils import sequences

tfd = tfp.distributions


def mean(_, x):
    return K.mean(x)


def fvm_entropy(kappa):
    """For d=3"""
    expk2 = K.exp(- 2 * kappa)
    return (
        1 + np.log(2 * np.pi)
        - 2 * kappa * expk2 / (1 - expk2)
        + tf.math.log1p(- expk2)
        - K.log(kappa)
    )


def mean_fvm_entropy(kappa):
    return K.mean(fvm_entropy(kappa))


def mean_neg_fvm_entropy(_, kappa):
    return - mean_fvm_entropy(kappa)


def fvm_cost(y_true, dist_pred):
    return - K.sum(dist_pred.mean() * y_true, axis=1)


def mean_fvm_cost(y_true, dist_pred):
    return K.mean(fvm_cost(y_true, dist_pred))


def mean_neg_log_prob(y_true, dist_pred):
    return - K.mean(dist_pred.log_prob(y_true))


def mean_neg_dot_prod(y_true, y_pred):
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return - K.mean(K.sum(y_true * y_pred, axis=1))


def neg_log_prob(y_true, dist_pred):
    return -dist_pred.log_prob(y_true)


def neg_dot_prod(y_true, y_pred):
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return - K.sum(y_true * y_pred, axis=1)


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
        

class Model(object):
    """docstring for Model"""
    
    def get_sequence(self, sample_path, batch_size, istraining=True):
        if sample_path is None:
            return None
        return getattr(sequences, self.sample_class)(sample_path,
                                                     batch_size,
                                                     istraining)

    @staticmethod
    def check(config):
        pass
        

class FvM(Model):
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


class FvMHybrid(Model):
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


class RNN(Model):

    model_name="RNN"

    sample_class = "RNNSamples"

    summaries = "RNNSummaries"

    def __init__(self, config):

        if 'input_shape' in config:
            input_shape = config['input_shape']
        else:
            input_shape = tuple(
                np.load(config["train_path"], allow_pickle=True)["input_shape"])

        batch_size = config["batch_size"]
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


class Entrack(Model):
    """docstring for Entrack"""
    model_name="Entrack"

    summaries = "EntrackSummaries"

    sample_class = "EntrackSamples"

    custom_objects = {
            "mean_fvm_cost": mean_fvm_cost,
            "mean_neg_fvm_entropy": mean_neg_fvm_entropy,
            "mean_neg_dot_prod": mean_neg_dot_prod,
            "kappa_mean": mean,
            "DistributionLambda": tfp.layers.DistributionLambda
        }

    def __init__(self, config):

        input_shape = tuple(
            np.load(config["train_path"], allow_pickle=True)["input_shape"])

        temperature = Temperature(config["temperature"])

        deep_update(config, {"temperature": temperature})

        inputs = Input(shape=input_shape, name="inputs")
        shared = self._shared_layers(inputs)
        kappa = self.kappa(shared)
        mu = self.mu(shared)

        self.keras = tf.keras.Model(
            inputs,
            [self.fvm(mu, kappa), kappa],
            name=self.model_name
        )

        self.temperature = temperature

    @staticmethod
    def _shared_layers(inputs):
        x = Dense(2048, activation="relu")(inputs)
        x = Dense(2048, activation="relu")(x)
        x = Dense(2048, activation="relu")(x)
        x = Dense(2048, activation="relu")(x)
        return x

    @staticmethod
    def kappa(x):
        kappa = Dense(1024, activation="relu")(x)
        kappa = Dense(1024, activation="relu")(kappa)
        kappa = Dense(1, activation="relu")(kappa)
        kappa = Lambda(lambda t: K.squeeze(t, 1) + 0.001, name="kappa")(kappa)
        return kappa
    
    @staticmethod
    def mu(x):
        mu = Dense(1024, activation="relu")(x)
        mu = Dense(1024, activation="relu")(mu)
        mu = Dense(3, activation="linear")(mu)
        mu = Lambda(lambda t: K.l2_normalize(t, axis=-1), name="mu")(mu)
        return mu

    @staticmethod
    def fvm(mu, kappa):
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
            loss={
                "fvm": self.custom_objects["mean_fvm_cost"],
                "kappa": self.custom_objects["mean_neg_fvm_entropy"]
            },
            loss_weights={"fvm": 1.0, "kappa": self.temperature},
            metrics={"fvm": self.custom_objects["mean_neg_dot_prod"],
                     "kappa": self.custom_objects["kappa_mean"]}
        )

    @staticmethod
    def check(config):
        """Assert model specific parameters"""
        assert "temperature" in config
        assert config["temperature"] > 0