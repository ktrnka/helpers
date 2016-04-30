from __future__ import print_function
from __future__ import unicode_literals

import unittest

import numpy
import sklearn.dummy
import sklearn.linear_model
import sklearn.metrics
import sklearn.pipeline

from .. import sk


class HelperTests(unittest.TestCase):
    def test_time_cross_validation_splitter(self):
        X, Y = _build_data(100)

        # regular test
        # TODO: This will fail if the list() is removed but it shouldn't
        splits = list(sk.TimeCV(X.shape[0], 4))
        self.assertSequenceEqual([(range(0, 50), range(50, 75)), (range(0, 75), range(75, 100))], splits)
        self.assertEqual(2, len(splits))

        # test with 2 buckets per test
        splits = list(sk.TimeCV(X.shape[0], 4, test_splits=2, balanced_tests=False))
        self.assertListEqual([(range(0, 50), range(50, 100)), (range(0, 75), range(75, 100))], splits)

        # test with no min training amount
        splits = list(sk.TimeCV(X.shape[0], 4, min_training=0))
        self.assertListEqual(
            [(range(0, 25), range(25, 50)), (range(0, 50), range(50, 75)), (range(0, 75), range(75, 100))], splits)

        splits = list(sk.TimeCV(49125, 10))
        print([(len(s), min(s), max(s), max(s) - min(s)) for _, s in splits])
        self.assertEqual(5, len(splits))

        for train, test in splits:
            self.assertEqual(4912, len(test))

    def test_get_name(self):
        model = sklearn.linear_model.LinearRegression()
        self.assertEqual("Linear", sk.get_model_name(model))
        self.assertEqual("LinearRegression", sk.get_model_name(model, remove=None))

        wrapped = sk.MultivariateRegressionWrapper(model)
        self.assertEqual("MultivariateWrapper(Linear)", sk.get_model_name(wrapped))

        # test bagging
        wrapped = sk.MultivariateBaggingRegressor(model)
        self.assertEqual("MultivariateBagging(Linear)", sk.get_model_name(wrapped))

        # test random search
        wrapped = sk.RandomizedSearchCV(model, None)
        self.assertEqual("RandomizedSearchCV(Linear)", sk.get_model_name(wrapped))

        # test a pipeline
        pipe = sklearn.pipeline.Pipeline([("lr", model)])
        self.assertEqual("Pipeline(Linear)", sk.get_model_name(pipe))
        self.assertEqual("Pipeline_Linear", sk.get_model_name(pipe, format="{}_{}"))


def _build_data(n):
    X = numpy.asarray(range(n))

    X = numpy.vstack((X, X + 1, X + 2, X + 3)).transpose()

    return X[:, :2], X[:, 2:]


class ModelTests(unittest.TestCase):
    def test_build_data(self):
        X, Y = _build_data(100)
        self.assertListEqual([100, 2], list(X.shape))
        self.assertListEqual([100, 2], list(Y.shape))

    def test_bagging(self):
        X, Y = _build_data(100)

        # test basic linear regression
        baseline_model = sklearn.linear_model.LinearRegression().fit(X, Y)

        Y_pred = baseline_model.predict(X)
        self.assertEqual(Y.shape, Y_pred.shape)
        baseline_error = sklearn.metrics.mean_squared_error(Y, Y_pred)
        self.assertLess(baseline_error, 1.)

        model = sk.MultivariateBaggingRegressor(base_estimator=sklearn.linear_model.LinearRegression(),
                                                max_samples=0.8, max_features=0.6)
        model.fit(X, Y)

        Y_pred = model.predict(X)
        self.assertEqual(Y.shape, Y_pred.shape)
        model_error = sklearn.metrics.mean_squared_error(Y, Y_pred)
        self.assertLess(model_error, 1.)

        # test that it's an improvement within some epsilon
        self.assertLessEqual(model_error, baseline_error + 1e-6)
