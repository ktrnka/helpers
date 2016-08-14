from __future__ import print_function
from __future__ import unicode_literals

import unittest

import numpy
import sklearn.dummy
import sklearn.linear_model
import sklearn.metrics
import sklearn.pipeline
import sklearn.ensemble

from .. import sk
from .. import feature_selection


class HelperTests(unittest.TestCase):
    def test_time_cross_validation_splitter(self):
        X, Y = _build_data(100)

        # regular test
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

    def test_wraparound_timecv(self):
        X, Y = _build_data(100)

        splits = list(sk.WraparoundTimeCV(X.shape[0], 5, 2))
        self.assertSequenceEqual([(range(60, 100), range(0, 20)), (range(80, 100) + range(0, 20), range(20, 40)), (range(0, 40), range(40, 60)), (range(20, 60), range(60, 80)), (range(40, 80), range(80, 100))], splits)
        self.assertEqual(5, len(splits))

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

    def test_joined_multi(self):
        """This test isn't good cause GradientBoosting really sucks at this kind of numerical prediction"""
        X, Y = _build_data(100)
        baseline_model = sklearn.linear_model.LinearRegression().fit(X, Y)

        Y_pred = baseline_model.predict(X)
        self.assertEqual(Y.shape, Y_pred.shape)
        baseline_error = sklearn.metrics.mean_squared_error(Y, Y_pred)
        self.assertLess(baseline_error, 1.)

        model = sk.JoinedMultivariateRegressionWrapper(sklearn.ensemble.GradientBoostingRegressor())
        model.fit(X, Y)
        Y_pred = model.predict(X)
        print(Y_pred)
        self.assertEqual(Y.shape, Y_pred.shape)
        model_error = sklearn.metrics.mean_squared_error(Y, Y_pred)
        self.assertLess(model_error, 1.)

    def test_joined_multi_order(self):
        X, Y = _build_data(100)
        Y = Y[:, 0]

        X2, Y2 = sk.JoinedMultivariateRegressionWrapper._rearrange(X, Y, 2)
        self.assertEqual(X[0, 0], X2[0, 0])
        self.assertEqual(X[50, 0], X2[1, 0])
        self.assertEqual(X[1, 0], X2[2, 0])
        self.assertEqual(X[51, 0], X2[3, 0])

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

    def test_delta_regressor_do_no_harm(self):
        X, Y = _build_data(100)

        baseline_model = sklearn.linear_model.LinearRegression().fit(X, Y)
        baseline_error = sklearn.metrics.mean_squared_error(Y, baseline_model.predict(X))

        model = sk.DeltaSumRegressor(baseline_model).fit(X, Y)
        error = sklearn.metrics.mean_squared_error(Y, model.predict(X))

        self.assertAlmostEqual(baseline_error, error, places=5)

    def test_delta_regressor_improvement(self):
        X, Y = _build_data(100)

        # make it quadratic so that the base can't fit it as well
        Y = Y ** 2

        baseline_model = sklearn.linear_model.LinearRegression().fit(X, Y)
        baseline_error = sklearn.metrics.mean_squared_error(Y, baseline_model.predict(X))

        model = sk.DeltaSumRegressor(baseline_model).fit(X, Y)
        error = sklearn.metrics.mean_squared_error(Y, model.predict(X))

        self.assertLess(error, baseline_error)


class TestFeatureSelectors(unittest.TestCase):
    def setUp(self):
        self.X, self.Y = _build_data(10)
        self.X[:, 0] = self.X[:, 0] ** 2  # break correlation so that FS picks 1 for unit test

    def test_loi(self):
        selector = feature_selection.LoiSelector(sklearn.linear_model.LinearRegression(), sk.rms_error)
        scores = selector.score(self.X, self.Y, sklearn.cross_validation.KFold(self.X.shape[0], n_folds=3))
        self.assertGreater(scores[1], scores[0])

    def test_loo(self):
        selector = feature_selection.LooSelector(sklearn.linear_model.LinearRegression(), sk.rms_error)
        scores = selector.score(self.X, self.Y, sklearn.cross_validation.KFold(self.X.shape[0], n_folds=3))
        self.assertGreater(scores[1], scores[0])

    def test_deform(self):
        selector = feature_selection.DeformSelector(sklearn.linear_model.LinearRegression(), sk.rms_error)
        scores = selector.score(self.X.astype(numpy.float32), self.Y, sklearn.cross_validation.KFold(self.X.shape[0], n_folds=3))
        self.assertGreater(scores[1], scores[0])
