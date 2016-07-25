from __future__ import unicode_literals
from __future__ import print_function

import time

import keras.backend
import keras.callbacks
import keras.constraints
import keras.layers
import keras.layers.recurrent
import keras.models
import keras.optimizers
import keras.regularizers
import numpy
import pandas
import sklearn
import sklearn.utils
import theano

from . import general

_he_activations = {"relu"}

# Normally it's always better to set this true but it only works if you edit Theano I think
RNN_UNROLL = True

def set_theano_float_precision(precision):
    assert precision in {"float32", "float64"}
    theano.config.floatX = precision


class NnRegressor(sklearn.base.BaseEstimator):
    """Wrapper for Keras feed-forward neural network for regression to enable scikit-learn grid search"""

    def __init__(self, hidden_layer_sizes=(100,), hidden_units=None, dropout=None, batch_size=-1, loss="mse",
                 num_epochs=500, activation="relu", input_noise=0., learning_rate=0.001, verbose=0, init=None, l2=None,
                 batch_norm=False, early_stopping=False, clip_gradient_norm=None, assert_finite=True,
                 maxnorm=False, val=0., history_file=None, optimizer="adam", input_dropout=None, lr_decay=None, non_negative=False, weight_samples=False):
        self.clip_gradient_norm = clip_gradient_norm
        self.assert_finite = assert_finite
        self.hidden_units = hidden_units
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.activation = activation
        self.input_noise = input_noise
        self.input_dropout = input_dropout
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.loss = loss
        self.l2 = l2
        self.batch_norm = batch_norm
        self.early_stopping = early_stopping
        self.init = self._get_default_init(init, activation)
        self.use_maxnorm = maxnorm
        self.val = val
        self.history_file = history_file
        self.optimizer = optimizer
        self.lr_decay = lr_decay
        self.extra_callback = None
        self.non_negative = non_negative
        self.weight_samples = weight_samples

        self.logger = general.get_class_logger(self)

        self.model_ = None
        self.history_df_ = None

    def _get_optimizer(self):
        if self.optimizer == "adam":
            return keras.optimizers.Adam(**self._get_optimizer_kwargs())
        elif self.optimizer == "rmsprop":
            return keras.optimizers.RMSprop(**self._get_optimizer_kwargs())
        elif self.optimizer == "sgd":
            return keras.optimizers.SGD(**self._get_optimizer_kwargs())
        elif self.optimizer == "adamax":
            return keras.optimizers.Adamax(**self._get_optimizer_kwargs())
        else:
            raise ValueError("Unknown optimizer {}".format(self.optimizer))

    def _get_activation(self):
        if self.activation == "elu":
            return keras.layers.advanced_activations.ELU()
        elif self.activation:
            return keras.layers.core.Activation(self.activation)
        else:
            raise ValueError("No activation unit specified")

    def fit(self, X, y, **kwargs):
        self.set_params(**kwargs)

        if self.hidden_units:
            self.hidden_layer_sizes = (self.hidden_units,)

        self.logger.debug("X: {}, Y: {}".format(X.shape, y.shape))

        model = keras.models.Sequential()

        # input noise not optional so that we have a well-defined first layer to
        # set the input shape on (though it may be set to zero noise)
        model.add(keras.layers.noise.GaussianNoise(self.input_noise, input_shape=X.shape[1:]))

        if self.input_dropout:
            model.add(keras.layers.core.Dropout(self.input_dropout))

        dense_kwargs = self._get_dense_layer_kwargs()

        # hidden layers
        for layer_size in self.hidden_layer_sizes:
            model.add(keras.layers.core.Dense(output_dim=layer_size, **dense_kwargs))
            if self.batch_norm:
                model.add(keras.layers.normalization.BatchNormalization())
            model.add(self._get_activation())

            if self.dropout:
                model.add(keras.layers.core.Dropout(self.dropout))

        # output layer
        model.add(keras.layers.core.Dense(output_dim=y.shape[1], **dense_kwargs))

        if self.non_negative:
            model.add(keras.layers.core.Activation("relu"))

        optimizer = self._get_optimizer()
        model.compile(loss=self.loss, optimizer=optimizer)

        self.model_ = model
        self._run_fit(X, y)

        return self

    def _run_fit(self, X, y):
        t = time.time()
        history = self.model_.fit(X, y, **self._get_fit_kwargs(X))
        t = time.time() - t

        self._save_history(history)

        self.logger.info("Trained at {:,} rows/sec in {:,} epochs".format(int(X.shape[0] * len(history.epoch) / t),
                                                                          len(history.epoch)))
        self.logger.debug("Model has {:,} params".format(self.count_params()))

    def _get_dense_layer_kwargs(self):
        """Apply settings to dense layer keyword args"""
        dense_kwargs = {"init": self.init, "trainable": True}
        if self.l2:
            dense_kwargs["W_regularizer"] = keras.regularizers.l2(self.l2)

        if self.use_maxnorm:
            dense_kwargs["W_constraint"] = keras.constraints.MaxNorm(2)
            dense_kwargs["b_constraint"] = keras.constraints.MaxNorm(2)

        return dense_kwargs

    def _get_fit_kwargs(self, X, batch_size_override=None, num_epochs_override=None, disable_validation=False):
        """Apply settings to the fit function keyword args"""
        kwargs = {"verbose": self.verbose, "nb_epoch": self.num_epochs, "callbacks": []}

        if num_epochs_override:
            kwargs["nb_epoch"] = num_epochs_override

        if self.early_stopping:
            monitor = "val_loss" if self.val > 0 else "loss"
            es = keras.callbacks.EarlyStopping(monitor=monitor, patience=self.num_epochs / 20, verbose=self.verbose,
                                               mode="min")
            kwargs["callbacks"].append(es)

        if self.lr_decay and self.lr_decay != 1:
            assert 0 < self.lr_decay < 1, "Learning rate must range 0-1"
            kwargs["callbacks"].append(LearningRateDecay(self.lr_decay))

        if self.extra_callback:
            kwargs["callbacks"].append(self.extra_callback)

        if self.val > 0 and not disable_validation:
            kwargs["validation_split"] = self.val

        if self.weight_samples:
            kwargs["sample_weight"] = 0.97 ** numpy.log(X.shape[0] - numpy.asarray(range(X.shape[0])))

        kwargs["batch_size"] = self.batch_size
        if batch_size_override:
            kwargs["batch_size"] = batch_size_override
        if kwargs["batch_size"] < 0 or kwargs["batch_size"] > X.shape[0]:
            kwargs["batch_size"] = X.shape[0]

        self.logger.info("Fit kwargs: %s", kwargs)

        return kwargs

    def count_params(self):
        return self.model_.count_params()

    def predict(self, X):
        retval = self._check_finite(self.model_.predict(X))
        return retval

    def _check_finite(self, Y):
        if self.assert_finite:
            sklearn.utils.assert_all_finite(Y)
        else:
            Y = numpy.nan_to_num(Y)

        return Y

    def _get_default_init(self, init, activation):
        if init:
            return init

        if activation in _he_activations:
            return "he_uniform"

        return "glorot_uniform"

    def _get_optimizer_kwargs(self):
        kwargs = {"lr": self.learning_rate}

        if self.clip_gradient_norm:
            kwargs["clipnorm"] = self.clip_gradient_norm

        return kwargs

    def _save_history(self, history):
        if not self.history_file:
            return

        self.history_df_ = pandas.DataFrame.from_dict(history.history)
        self.history_df_.index.rename("epoch", inplace=True)
        self.history_df_.to_csv(self.history_file)


def pad_to_batch(data, batch_size):
    remainder = data.shape[0] % batch_size
    if remainder == 0:
        return data, lambda Y: Y

    pad_after = [batch_size - remainder] + [0 for _ in data.shape[1:]]
    paddings = [(0, p) for p in pad_after]
    return numpy.pad(data, paddings, mode="edge"), lambda Y: Y[:-pad_after[0]]


class RnnRegressor(NnRegressor):
    def __init__(self, num_units=50, time_steps=5, batch_size=100, num_epochs=100, unit="lstm", verbose=0,
                 early_stopping=False, dropout=None, recurrent_dropout=None, loss="mse", input_noise=0.,
                 learning_rate=0.001, clip_gradient_norm=None, val=0, assert_finite=True, history_file=None,
                 pretrain=True, optimizer="adam", input_dropout=None, activation=None, posttrain=False, hidden_layer_sizes=None, stateful=False,
                 lr_decay=None, non_negative=False, l2=None, reverse=False):
        super(RnnRegressor, self).__init__(batch_size=batch_size, num_epochs=num_epochs, verbose=verbose,
                                           early_stopping=early_stopping, dropout=dropout, loss=loss,
                                           input_noise=input_noise, learning_rate=learning_rate,
                                           clip_gradient_norm=clip_gradient_norm, val=val, assert_finite=assert_finite,
                                           history_file=history_file, optimizer=optimizer, input_dropout=input_dropout,
                                           activation=activation, hidden_layer_sizes=hidden_layer_sizes, lr_decay=lr_decay, non_negative=non_negative, l2=l2)

        self.posttrain = posttrain
        self.num_units = num_units
        self.time_steps = time_steps
        self.unit = unit
        self.recurrent_dropout = recurrent_dropout
        self.use_maxnorm = True
        self.pretrain = pretrain
        self.stateful = stateful
        self.reverse = reverse

        if stateful:
            assert self.time_steps == self.batch_size

        self.logger = general.get_class_logger(self)

    def _transform_input(self, X):
        return general.prepare_time_matrix(X, self.time_steps, fill_value=0)

    def _get_recurrent_layer_kwargs(self):
        """Apply settings to dense layer keyword args"""
        kwargs = {"output_dim": self.num_units, "trainable": True, "unroll": RNN_UNROLL}

        if self.recurrent_dropout:
            kwargs["dropout_U"] = self.recurrent_dropout

        if self.l2:
            kwargs["W_regularizer"] = keras.regularizers.l2(self.l2)
            kwargs["U_regularizer"] = keras.regularizers.l2(self.l2)


        return kwargs

    def _check_reverse(self, *args):
        """Return the args unless the reverse flag is set, then reverse all the matrices"""
        if len(args) == 1:
            if self.reverse:
                return args[0][::-1]
            else:
                return args[0]
        else:
            if self.reverse:
                return [arg[::-1] for arg in args]
            else:
                return args

    def fit(self, X, Y, **kwargs):
        self.set_params(**kwargs)

        X, Y = self._check_reverse(X, Y)

        model = keras.models.Sequential()

        X_time = self._transform_input(X)

        if self.stateful:
            X_time, _ = pad_to_batch(X_time, self.batch_size)
            Y, _ = pad_to_batch(Y, self.batch_size)

        self.logger.debug("X takes %d mb", X.nbytes / 10e6)
        self.logger.debug("X_time takes %d mb", X_time.nbytes / 10e6)

        if self.stateful:
            model.add(keras.layers.noise.GaussianNoise(self.input_noise, batch_input_shape=(self.batch_size,) + X_time.shape[1:]))
        else:
            model.add(keras.layers.noise.GaussianNoise(self.input_noise, input_shape=X_time.shape[1:]))

        if self.input_dropout:
            model.add(keras.layers.core.Dropout(self.input_dropout))

        # recurrent layer
        if self.unit == "lstm":
            model.add(keras.layers.recurrent.LSTM(**self._get_recurrent_layer_kwargs()))
        elif self.unit == "gru":
            model.add(keras.layers.recurrent.GRU(**self._get_recurrent_layer_kwargs()))
        else:
            raise ValueError("Unknown unit type: {}".format(self.unit))

        # dropout
        if self.dropout:
            model.add(keras.layers.core.Dropout(self.dropout))

        # regular hidden layer(s)
        if self.hidden_layer_sizes:
            for layer_size in self.hidden_layer_sizes:
                self.logger.warning("Adding FC-%d layer after RNN", layer_size)

                model.add(keras.layers.core.Dense(output_dim=layer_size, **self._get_dense_layer_kwargs()))
                model.add(self._get_activation())

                # if self.dropout:
                #     model.add(keras.layers.core.Dropout(self.dropout))

        # output layer
        model.add(keras.layers.core.Dense(output_dim=Y.shape[1], **self._get_dense_layer_kwargs()))
        if self.non_negative:
            model.add(keras.layers.core.Activation("relu"))

        optimizer = self._get_optimizer()
        model.compile(loss="mse", optimizer=optimizer)
        self.model_ = model

        if self.pretrain and not self.stateful:
            self.model_.fit(X_time, Y, **self._get_fit_kwargs(X, batch_size_override=1, num_epochs_override=1))

        self._run_fit(X_time, Y)

        if self.posttrain and not self.stateful:
            self.model_.fit(X_time, Y, **self._get_fit_kwargs(X, disable_validation=True, num_epochs_override=5))

        return self

    def _get_fit_kwargs(self, *args, **kwargs):
        kwargs = super(RnnRegressor, self)._get_fit_kwargs(*args, **kwargs)

        if self.stateful:
            kwargs["shuffle"] = False
            kwargs["validation_split"] = 0

            self.logger.warning("Disabling validation split for stateful RNN training")

        return kwargs

    def predict(self, X):
        X = self._check_reverse(X)
        inverse_trans = None
        if self.stateful:
            self.model_.reset_states()
            X, inverse_trans = pad_to_batch(X, self.batch_size)
        Y = self._check_finite(self.model_.predict(self._transform_input(X)))

        if inverse_trans:
            Y = inverse_trans(Y)

        Y = self._check_reverse(Y)
        return Y


def make_learning_rate_schedule(initial_value, exponential_decay=0.99, kick_every=10000):
    logger = general.get_function_logger()

    def schedule(epoch_num):
        lr = initial_value * (10 ** int(epoch_num / kick_every)) * exponential_decay ** epoch_num
        logger.debug("Setting learning rate at {} to {}".format(epoch_num, lr))
        return lr

    return schedule


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


class AdaptiveLearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler that increases or decreases LR based on a recent sample of validation results"""

    def __init__(self, initial_learning_rate, monitor="val_loss", scale=2., window=5):
        super(AdaptiveLearningRateScheduler, self).__init__()
        self.monitor = monitor
        self.initial_lr = initial_learning_rate
        self.scale = float(scale)
        self.window = window

        self.metric_ = []
        self.logger_ = general.get_class_logger(self)

    def on_epoch_begin(self, epoch, logs={}):
        assert hasattr(self.model.optimizer, 'lr'), 'Optimizer must have a "lr" attribute.'

        lr = self._get_learning_rate()

        if lr:
            self.logger_.debug("Setting learning rate at %d to %e", epoch, lr)
            keras.backend.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs={}):
        metric = logs[self.monitor]
        self.metric_.append(metric)

    def _get_learning_rate(self):
        if len(self.metric_) < self.window * 2:
            return self.initial_lr

        data = numpy.asarray(self.metric_)

        baseline = data[:-self.window].min()
        diffs = baseline - data[-self.window:]

        # assume error, lower is better
        percent_epochs_improved = sigmoid((diffs / baseline) / 0.02).mean()
        self.logger_.debug("Ratio of good epochs: %.2f", percent_epochs_improved)

        if percent_epochs_improved > 0.75:
            return self._scale_learning_rate(self.scale)
        elif percent_epochs_improved < 0.5:
            return self._scale_learning_rate(1. / self.scale)

        return None

    def _scale_learning_rate(self, scale):
        return keras.backend.get_value(self.model.optimizer.lr) * scale


class LearningRateDecay(keras.callbacks.Callback):
    """Trying to get mode debug info...."""
    def __init__(self, decay):
        super(LearningRateDecay, self).__init__()
        self.decay = decay
        self.logger = general.get_class_logger(self)

    def on_epoch_end(self, epoch, logs={}):
        lr = keras.backend.get_value(self.model.optimizer.lr)
        self.logger.debug("Decay LR to {}".format(lr * self.decay))
        keras.backend.set_value(self.model.optimizer.lr, lr * self.decay)
