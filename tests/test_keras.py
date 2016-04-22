from __future__ import print_function
from __future__ import unicode_literals

import unittest

import math

import numpy
import sklearn.pipeline
import sklearn.linear_model
import sklearn.metrics
import sklearn.dummy

from .. import neural
from .test_sk import _build_data


def _build_periodic_data(n, period=50.):
    X = numpy.asarray(range(n), dtype=numpy.float32)

    X = numpy.vstack((X, numpy.cos(2. * math.pi * X / period))).transpose()
    return X[:, 0].reshape((-1, 1)), X[:, 1].reshape((-1, 1))


def moving_average(a, n=3):
    """From http://stackoverflow.com/a/14314054/1492373"""
    ret = numpy.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def _build_summation_data(n, lag=4):
    X = (numpy.random.rand(n, 1) > 0.5).astype(numpy.int16)

    Y = moving_average(X, lag).reshape(-1, 1)

    return X[lag - 1:, :], Y


def _build_summation_data_for_stateful(n, lag=4):
    X = (numpy.random.rand(n, 1) > 0.5).astype(numpy.int16)

    Y = moving_average(X, lag).reshape(-1, 1)

    return X[lag - 1:, :], numpy.hstack([Y, numpy.roll(Y, lag, axis=0)])


def _build_identity(n):
    X = numpy.random.rand(n, 1)
    return X, X


class ModelTests(unittest.TestCase):
    def test_nn_identity(self):
        X, Y = _build_identity(100)

        baseline_model = sklearn.dummy.DummyRegressor("mean").fit(X, Y)
        baseline_error = sklearn.metrics.mean_squared_error(Y, baseline_model.predict(X))

        nn = neural.NnRegressor(learning_rate=0.05, num_epochs=200, hidden_units=5, verbose=1)
        nn.fit(X, Y)
        nn_error = sklearn.metrics.mean_squared_error(Y, nn.predict(X))

        # should fit better than baseline
        self.assertLess(nn_error, baseline_error)
        self.assertLess(nn_error, baseline_error / 100)

        # should be able to fit the training data completely (but doesn't, depending on the data)
        self.assertAlmostEqual(0, nn_error, places=4)

    def test_nn_regression_model(self):
        # TODO: Replace this with Boston dataset or something
        X, Y = _build_data(100)

        model = neural.NnRegressor(learning_rate=0.01, num_epochs=1000, hidden_units=3)
        model.fit(X, Y)

        Y_pred = model.predict(X)
        self.assertEqual(Y.shape, Y_pred.shape)
        error = ((Y - Y_pred) ** 2).mean().mean()
        self.assertLess(error, 1.)

    def test_rnn(self):
        X, Y = _build_summation_data(1000, lag=4)

        # baseline linear regression
        baseline_model = sklearn.linear_model.LinearRegression().fit(X, Y)

        baseline_predictions = baseline_model.predict(X)
        self.assertEqual(Y.shape, baseline_predictions.shape)
        baseline_error = sklearn.metrics.mean_squared_error(Y, baseline_predictions)
        self.assertLess(baseline_error, 0.1)

        # test non-RNN
        model = neural.NnRegressor(activation="tanh", batch_size=50, num_epochs=100, verbose=0, early_stopping=True)
        model.fit(X, Y)
        mlp_predictions = model.predict(X)
        self.assertEqual(Y.shape, mlp_predictions.shape)
        mlp_error = ((Y - mlp_predictions) ** 2).mean().mean()
        self.assertLess(mlp_error, baseline_error * 1.2)

        # test RNN
        model = neural.RnnRegressor(num_epochs=200, batch_size=50, num_units=50, time_steps=5, early_stopping=True)
        model.fit(X, Y)
        rnn_predictions = model.predict(X)

        self.assertEqual(Y.shape, rnn_predictions.shape)
        error = ((Y - rnn_predictions) ** 2).mean().mean()

        print("RNN error", error)

        # should be more than 10x better
        self.assertLessEqual(error, mlp_error / 10)

    def test_build_data(self):
        X, Y = _build_data(100)
        self.assertListEqual([100, 2], list(X.shape))
        self.assertListEqual([100, 2], list(Y.shape))

    def test_learning_rate_scheduler(self):
        get_rate = neural.make_learning_rate_schedule(0.01, exponential_decay=0.995)

        self.assertAlmostEqual(get_rate(50), 0.007783125570686419)
        self.assertAlmostEqual(get_rate(200), 0.00366957821726167)

        get_kicking_rate = neural.make_learning_rate_schedule(0.01, exponential_decay=0.995, kick_every=100)
        self.assertAlmostEqual(get_kicking_rate(50), 0.007783125570686419)
        self.assertAlmostEqual(get_kicking_rate(101), get_rate(101) * 10)
        self.assertAlmostEqual(get_kicking_rate(201), get_rate(201) * 100)

    def test_stateful_padding(self):
        X = numpy.random.rand(18, 6)
        X2, inverter = neural.pad_to_batch(X, 5)

        print("X", X.shape)
        print("X2", X2.shape)
        self.assertEqual(X.shape[1], X2.shape[1])
        self.assertEqual(20, X2.shape[0])

        self.assertTrue(numpy.array_equal(X, X2[:X.shape[0]]))

        for row in range(X.shape[0], X2.shape[0]):
            self.assertTrue(numpy.array_equal(X[-1], X2[row]))

        self.assertTrue(numpy.array_equal(X, inverter(X2)))

    def test_stateful_rnn(self):
        X, Y = _build_summation_data(1000, lag=10)

        # baseline linear regression
        baseline_model = sklearn.linear_model.LinearRegression().fit(X, Y)

        baseline_predictions = baseline_model.predict(X)
        self.assertEqual(Y.shape, baseline_predictions.shape)
        baseline_error = sklearn.metrics.mean_squared_error(Y, baseline_predictions)
        self.assertLess(baseline_error, 0.1)

        # test plain RNN
        model = neural.RnnRegressor(num_epochs=200, batch_size=10, num_units=50, time_steps=10, early_stopping=True, verbose=1)
        model.fit(X, Y)
        vanilla_predictions = model.predict(X)

        self.assertEqual(Y.shape, vanilla_predictions.shape)
        vanilla_error = sklearn.metrics.mean_squared_error(Y, vanilla_predictions)

        print("Vanilla RNN error", vanilla_error)

        # should be more than 10x better
        self.assertLessEqual(vanilla_error, baseline_error / 10)

        # test stateful RNN
        model = neural.RnnRegressor(num_epochs=200, batch_size=10, num_units=50, time_steps=10, early_stopping=True, stateful=True)
        model.fit(X, Y)
        stateful_predictions = model.predict(X)

        self.assertEqual(Y.shape, stateful_predictions.shape)
        error = sklearn.metrics.mean_squared_error(Y, stateful_predictions)

        print("Stateful RNN error", error)

        # should be more than 1.5x better
        self.assertLessEqual(error, vanilla_error / 1.5)
