from __future__ import print_function
from __future__ import unicode_literals

import sys
import unittest
from operator import itemgetter

import numpy
import pandas
import sklearn
import sklearn.base
import sklearn.cross_validation
import sklearn.datasets
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics.scorer
import sklearn.utils.validation
from sklearn.metrics import mean_absolute_error

from . import general

"""
Data is weird? Model is doing weird shit? This module helps but only if you have beer.
"""


def get_input_gradient(y_true, y_pred, x, model, eps=1e-4, elementwise_eps=None, max_eps=20.):
    """Numeric gradient on this one example with respect to the inputs (assumes the model weights are fixed)"""
    assert isinstance(x, numpy.ndarray)
    if elementwise_eps is None:
        elementwise_eps = numpy.ones_like(x) * eps

    # MSE, should be a single number
    base_error = ((y_true - y_pred) ** 2).mean()

    # build a perturbation matrix of size the number of elements
    pos_shifted = stack_all_perturbations(x, 0, elementwise_scale=elementwise_eps)
    assert len(pos_shifted.shape) == 2

    perturbed_predictions = model.predict(pos_shifted)

    # for univariate some models will output a vector rather than 2d array so we need to fix that
    if len(perturbed_predictions.shape) == 1:
        perturbed_predictions = perturbed_predictions.reshape(-1, 1)

    # MSE for each row of ppredictions
    perturbed_error = ((y_true - perturbed_predictions) ** 2).mean(axis=1)

    d_error = (perturbed_error - base_error).flatten()
    d_x = x * elementwise_eps / 2.

    input_gradients = d_error / d_x

    # if any gradients are zero it could be that our eps is too small
    if any(grad == 0 for grad in input_gradients):
        scale = (input_gradients == 0).astype(numpy.float32) * 10 + 1

        wider_eps = numpy.minimum(scale * elementwise_eps, numpy.ones_like(scale) * max_eps)

        if not numpy.array_equal(elementwise_eps, wider_eps):
            return get_input_gradient(y_true, y_pred, x, model, elementwise_eps=wider_eps, max_eps=max_eps)

    return input_gradients


def compute_input_fairness(data_train, data_test, verbose=0, names=None):
    assert isinstance(data_train, numpy.ndarray)
    assert isinstance(data_test, numpy.ndarray)

    mean = data_train.mean(axis=0)
    std = data_train.std(axis=0)

    z_scores = (data_test - mean) / std
    mean_z_scores = z_scores.mean(axis=0)

    abs_z_scores = abs(z_scores)
    mean_abs_z_scores = abs_z_scores.mean(axis=0)

    if verbose:
        print("Average abs z-score of test w.r.t train: ", mean_abs_z_scores.mean())

    # compute percent out of some range like 5 sigma
    outliers = abs(z_scores) > 5
    outlier_percents = outliers.mean(axis=0)

    if verbose:
        for i, (feature_percentage, z_score, abs_z) in enumerate(zip(outlier_percents, mean_z_scores, mean_abs_z_scores)):
            name = names[i] if names is not None else i

            if feature_percentage > 0.01:
                print("Feature {}: {:.1f}% outliers, z: {:.3f}, abs(z): {:.3f}".format(name, 100. * feature_percentage, z_score, abs_z))

    return mean_abs_z_scores.mean(), max(mean_abs_z_scores), max(outlier_percents)


def compute_cross_validation_fairness(X, X_names, Y, Y_names, splits):
    for i, (train, test) in enumerate(splits):
        print("Checking fairness of CV split {}".format(i))

        mean_mean_abs_z, max_mean_abs_z, max_outliers = compute_input_fairness(X[train], X[test])
        if mean_mean_abs_z > 1. or max_mean_abs_z > 1. or max_outliers > 0.05:
            print("X[test] doesn't match X[train] well", mean_mean_abs_z, max_mean_abs_z, max_outliers)
            compute_input_fairness(X[train], X[test], verbose=1, names=X_names)

        mean_mean_abs_z, max_mean_abs_z, max_outliers = compute_input_fairness(Y[train], Y[test])
        if mean_mean_abs_z > 1. or max_mean_abs_z > 1. or max_outliers > 0.05:
            print("Y[test] doesn't match Y[train] well", mean_mean_abs_z, max_mean_abs_z, max_outliers)
            compute_input_fairness(Y[train], Y[test], verbose=1, names=Y_names)


def cross_val_score(estimator, X, y=None, scoring=None, cv=None):
    """Run cross-validation like normal but return (scores, predictions)"""
    ### TODO: Test this code. It's 100% untested!
    X, y = sklearn.utils.validation.indexable(X, y)

    cv = sklearn.cross_validation.check_cv(cv, X, y, classifier=sklearn.base.is_classifier(estimator))
    scorer = sklearn.metrics.scorer.check_scoring(estimator, scoring=scoring)

    y_pred = numpy.zeros_like(y)

    scores = []
    for train, test in cv:
        current_est = sklearn.base.clone(estimator).fit(X[train], y[train])
        predictions = current_est.predict(X[test])
        scores.append(scorer(y[test], predictions))
        y_pred[test] = predictions

    return numpy.asarray(scores), y_pred


def explain_input_gradient(input_gradient):
    for i, derivative in enumerate(input_gradient):
        print("To get desired prediction, the model wants you to adjust feature {} by {}".format(i, -derivative))


def stack_all_perturbations(x, global_scale, elementwise_scale=None, dtype=numpy.float32):
    """If x is a vector, this generates a square matrix and we perturb the matrix along the diagonal"""
    perturbations = numpy.tile(x.reshape(1, -1), (x.shape[0], 1)).astype(dtype)

    assert perturbations.shape[0] == perturbations.shape[1]

    perturb_matrix = numpy.ones_like(perturbations)
    if global_scale:
        perturb_matrix += numpy.identity(perturb_matrix.shape[0]) * global_scale
    if elementwise_scale is not None:
        perturb_matrix += numpy.diag(elementwise_scale)

    return perturbations * perturb_matrix


class EvaluationTests(unittest.TestCase):
    def test_fairness(self):
        X, y = sklearn.datasets.make_regression(n_samples=5000, n_features=20)
        X_train, X_test = sklearn.cross_validation.train_test_split(X, train_size=0.2)
        mean_z, max_abs_z, max_outlier_fraction = compute_input_fairness(X_train, X_test)
        self.assertLess(mean_z, 1.)
        self.assertLess(max_abs_z, 1.)
        self.assertLess(max_outlier_fraction, 0.01)

        # test on differently distributed data
        mean_z, max_abs_z, max_outlier_fraction = compute_input_fairness(X, X + 20 * numpy.random.rand(*X.shape) - 10)
        self.assertGreater(mean_z, 2.)
        self.assertGreater(max_abs_z, 2.)
        self.assertGreater(max_outlier_fraction, 0.1)


class ExplanationTests(unittest.TestCase):
    @staticmethod
    def _make_data(n, num_outputs=1):
        ones = numpy.ones((n, 1))
        X = numpy.hstack([numpy.cumsum(ones).reshape(ones.shape), ones])

        Y = numpy.tile(X[:, 0], (num_outputs, 1)).transpose()
        Y += numpy.random.rand(*Y.shape)

        return X, Y

    def test_perturbation(self):
        row = numpy.asarray([1, 2, 3])

        perturbations = stack_all_perturbations(row, .1)

        for i in range(len(row)):
            for j in range(len(row)):
                if i == j:
                    self.assertAlmostEqual(1.1 * row[i], perturbations[i, j], places=5)
                else:
                    self.assertEqual(row[j], perturbations[i, j])

    def test_explain_linear_regression(self):
        X, Y = self._make_data(100)
        X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, Y, test_size=0.2)

        # train a model, no intercept so that it's easier to diagnose
        model = sklearn.linear_model.LinearRegression(fit_intercept=False).fit(X_train, Y_train)
        predictions = model.predict(X_test)

        # check that it has a semi-ok fit
        self.assertLess(mean_absolute_error(Y_test, predictions), 10)

        # now diagnose all examples - it should be weighting the first feature the most
        for i in range(Y_test.shape[0]):
            input_gradient = get_input_gradient(Y_test[i], predictions[i], X_test[i], model)
            self.assertLess(abs(input_gradient[1]), abs(input_gradient[0]))

    def test_explain_multivariate(self):
        X, Y = self._make_data(100, 2)
        X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, Y, test_size=0.2)

        # train a model, no intercept so that it's easier to diagnose
        model = sklearn.linear_model.LinearRegression(fit_intercept=False).fit(X_train, Y_train)
        predictions = model.predict(X_test)

        # check that it has a semi-ok fit
        self.assertLess(mean_absolute_error(Y_test, predictions), 10)

        # now diagnose all examples - it should be weighting the first feature the most
        for i in range(Y_test.shape[0]):
            input_gradient = get_input_gradient(Y_test[i], predictions[i], X_test[i], model)

            self.assertLess(abs(input_gradient[1]), abs(input_gradient[0]))

    def test_explain_random_forest(self):
        X, Y = self._make_data(100)
        X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, Y, test_size=0.2)

        model = sklearn.ensemble.RandomForestRegressor(5, max_depth=5, random_state=4).fit(X_train, Y_train)
        predictions = model.predict(X_test)

        # check that it has a semi-ok fit
        self.assertLess(mean_absolute_error(Y_test, predictions), 10)

        # now diagnose all examples - it should be weighting the first feature the most
        for i in range(Y_test.shape[0]):
            input_gradient = get_input_gradient(Y_test[i], predictions[i], X_test[i], model)
            self.assertLess(abs(input_gradient[1]), abs(input_gradient[0]))


def verify_splits(X, Y, splits):
    for i, (train, test) in enumerate(splits):
        # analyse train and test
        print("Split {}".format(i))

        print("\tX[train].mean diff: ", X[train].mean(axis=0) - X.mean(axis=0))
        print("\tX[train].std diffs: ", X[train].std(axis=0) - X.std(axis=0))
        print("\tY[train].mean: ", Y[train].mean(axis=0))
        print("\tY[train].std: ", Y[train].std(axis=0).mean())


def verify_data(X_df, Y_df, filename):
    assert isinstance(X_df, pandas.DataFrame)
    assert isinstance(Y_df, pandas.DataFrame)
    logger = general.get_function_logger()

    # check NaN inputs
    data_na = X_df.isnull().sum(axis=1)
    if data_na.sum() > 0:
        logger.error("Null values in feature matrix")

        for feature, na_count in data_na.iteritems():
            if na_count > 0:
                logger.error("{}: {:.1f}% null ({:,} / {:,})".format(feature, 100. * na_count / len(X_df), na_count, len(X_df)))

        sys.exit(-1)

    # test for zero stddev
    train_std = X_df.std()
    for feature, std in train_std.iteritems():
        if std == 0:
            logger.warning("{} has std dev zero".format(feature))

    # scale both input and output
    X = sklearn.preprocessing.RobustScaler().fit_transform(X_df)
    Y = sklearn.preprocessing.StandardScaler().fit_transform(Y_df)

    # find rows with values over 10x IQR from median
    train_deviants = numpy.abs(X) > 10
    train_deviant_rows = train_deviants.sum(axis=1) > 0
    deviant_feature_counts = train_deviants.sum(axis=0)

    logger.warn("Input has {:,} values beyond 10x IQR".format(deviant_feature_counts.sum()))
    for feature, deviant_count in sorted(zip(X_df.columns, deviant_feature_counts), key=itemgetter(1), reverse=True):
        if deviant_count == 0:
            break

        logger.info("Input {}: {:,} values beyond 10x IQR".format(feature, deviant_count))

    deviant_df = pandas.DataFrame(numpy.hstack([X[train_deviant_rows], Y[train_deviant_rows]]), columns=list(X_df.columns) + list(Y_df.columns))
    if deviant_df.shape[0] > 0:
        logger.warn("Found {:,} deviant rows".format(deviant_df.shape[0]))

        if filename:
            logger.warn("Saving deviant rows to {}".format(filename))
            deviant_df.to_csv(filename)
    else:
        print("No deviant rows")

    # check for outliers in outputs
    Y = sklearn.preprocessing.RobustScaler().fit_transform(Y_df)
    Y_deviants = numpy.abs(Y) > 10
    deviant_rows = Y_deviants.sum(axis=1)
    deviant_cols = Y_deviants.sum(axis=0)

    if deviant_rows.sum() > 0:
        logger.warn("Output has {:,} values beyond 10x IQR".format(deviant_rows.sum()))

        for feature, deviant_count in sorted(zip(Y_df.columns, deviant_cols), key=itemgetter(1), reverse=True):
            if deviant_count == 0:
                break

            logger.info("{}: {:,} values beyond 10x IQR".format(feature, deviant_count))
