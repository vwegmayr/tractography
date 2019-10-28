import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Input, Reshape, Dropout,
    BatchNormalization, Lambda, Dense)

tfd = tfp.distributions


def mean_neg_log_prob(y_true, pred_dist):
    return -K.mean(pred_dist.log_prob(y_true))

def mean_neg_dot_prod(y_true, y_pred):
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.mean(K.sum(y_true * y_pred, axis=1))

def neg_log_prob(y_true, pred_dist):
    return -pred_dist.log_prob(y_true)

def neg_dot_prod(y_true, y_pred):
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.sum(y_true * y_pred, axis=1)


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
        
    def __init__(self, input_shape, loss_weight=None):

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

    def __init__(self, input_shape, loss_weight):

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
        kappa = Lambda(lambda t: K.squeeze(t, 1), name="kappa")(kappa)

        fvm = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda params: tfd.VonMisesFisher(
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