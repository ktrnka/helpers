from __future__ import print_function

"""
Data is weird? Model is doing weird shit? This module helps but only if you have beer.
"""

from __future__ import unicode_literals
import unittest
import numpy

import sklearn.linear_model
import sklearn.cross_validation
from sklearn.metrics import mean_absolute_error
import sklearn.ensemble
import sklearn.datasets
import sklearn.utils.validation
import sklearn.base
import sklearn.metrics.scorer

def get_input_gradient(y_true, y_pred, x, model, eps=1e-4, param_eps=None, max_eps=20.):
    """Approximate numeric gradient on this one example with respect to the inputs (assumes the model weights are fixed)"""
    assert isinstance(x, numpy.ndarray)
    if param_eps is None:
        param_eps = numpy.ones_like(x) * eps

    # MSE, should be a single number
    base_error = ((y_true - y_pred) ** 2).mean()

    # build a perturbation matrix of size the number of elements
    pos_shifted = stack_all_perturbations(x, 0, perturb_elements=param_eps)
    assert len(pos_shifted.shape) == 2

    perturbed_predictions = model.predict(pos_shifted)

    # for univariate some models will output a vector rather than 2d array so we need to fix that
    if len(perturbed_predictions.shape) == 1:
        perturbed_predictions = perturbed_predictions.reshape(-1, 1)

    # MSE for each row of ppredictions
    perturbed_error = ((y_true - perturbed_predictions) ** 2).mean(axis=1)

    d_error = (perturbed_error - base_error).flatten()
    d_x = x * param_eps / 2.

    input_gradients = d_error / d_x

    # if any gradients are zero it could be that our eps is too small
    if any(grad == 0 for grad in input_gradients):
        scale = (input_gradients == 0).astype(numpy.float32) * 10 + 1

        wider_eps = numpy.minimum(scale * param_eps, numpy.ones_like(scale) * max_eps)

        if not numpy.array_equal(param_eps, wider_eps):
            return get_input_gradient(y_true, y_pred, x, model, param_eps=wider_eps, max_eps=max_eps)

    return input_gradients


def compute_input_fairness(data_train, data_test, verbose=0):
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
            print("Feature {}: {:.1f}% outliers, z: {:.3f}, abs(z): {:.3f}".format(i, 100. * feature_percentage, z_score, abs_z))

    return mean_abs_z_scores.mean(), max(mean_abs_z_scores), max(outlier_percents)


def compute_cross_validation_fairness(X, Y, splits):
    ### TODO: Test this code. It's 100% untested!
    for i, (train, test) in enumerate(splits):
        print("Checking fairness of CV split {}".format(i))

        mean_mean_abs_z, max_mean_abs_z, max_outliers = compute_input_fairness(X[train], X[test])
        if mean_mean_abs_z > 1. or max_mean_abs_z > 1. or max_outliers > 0.05:
            print("X{test] doesn't match X[train] well", mean_mean_abs_z, max_mean_abs_z, max_outliers)
            compute_input_fairness(X[train], X[test], verbose=1)

        mean_mean_abs_z, max_mean_abs_z, max_outliers = compute_input_fairness(Y[train], Y[test])
        if mean_mean_abs_z > 1. or max_mean_abs_z > 1. or max_outliers > 0.05:
            print("Y{test] doesn't match Y[train] well", mean_mean_abs_z, max_mean_abs_z, max_outliers)
            compute_input_fairness(Y[train], Y[test], verbose=1)


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


def stack_all_perturbations(x, perturb_all, perturb_elements=None, dtype=numpy.float32):
    """If x is a vector, this generates a square matrix and we perturb the matrix along the diagonal"""
    perturbations = numpy.tile(x.reshape(1, -1), (x.shape[0], 1)).astype(dtype)

    assert perturbations.shape[0] == perturbations.shape[1]

    perturb_matrix = numpy.ones_like(perturbations)
    if perturb_all:
        perturb_matrix += numpy.identity(perturb_matrix.shape[0]) * perturb_all
    if perturb_elements is not None:
        perturb_matrix += numpy.diag(perturb_elements)

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

        for i in xrange(len(row)):
            for j in xrange(len(row)):
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
        for i in xrange(Y_test.shape[0]):
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
        for i in xrange(Y_test.shape[0]):
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
        for i in xrange(Y_test.shape[0]):
            input_gradient = get_input_gradient(Y_test[i], predictions[i], X_test[i], model)
            self.assertLess(abs(input_gradient[1]), abs(input_gradient[0]))
