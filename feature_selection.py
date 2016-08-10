from __future__ import print_function
from __future__ import unicode_literals

import scipy.stats
import sklearn.ensemble

"""
Helpers for feature selection. Not entirely generalized from Mars Express Challenge code.
"""

import numpy
import sklearn.feature_selection
import sklearn.linear_model


def score_features_elastic_net(X, Y, alpha=0.001):
    """ElasticNet coefs against summed Y to simplify multivariate"""
    model = sklearn.linear_model.ElasticNet(alpha)
    model.fit(X, Y.sum(axis=1))
    return sklearn.feature_selection.from_model._get_feature_importances(model)


def score_features_ridge(X, Y, alpha=0.01):
    model = sklearn.linear_model.Ridge(alpha)
    model.fit(X, Y.sum(axis=1))
    return sklearn.feature_selection.from_model._get_feature_importances(model)


class BaseFeatureSelector(object):
    def __init__(self):
        pass

    def score(self, X, Y):
        pass

    def cv_score(self, X, Y, splits, stddev_weight=0.05):
        score_matrix = []
        for train, test in splits:
            score_matrix.append(self.score(X[train], Y[train]))

        score_matrix = numpy.asarray(score_matrix)
        feature_scores = score_matrix.mean(axis=0) + stddev_weight * score_matrix.std(axis=0)
        return feature_scores


def score_features_random_forest(X, Y, n_jobs=-1):
    """Random forest feature importances, supports multivariate"""
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=100, n_jobs=n_jobs)
    model.fit(X, Y)
    return sklearn.feature_selection.from_model._get_feature_importances(model)


def inverse_rank_order(weights, base=0.9):
    """Inverse rank order, can be used to bring different feature scores onto the same scale but it didn't work well for me"""
    return base ** scipy.stats.rankdata(weights)


def multivariate_select(X, Y, univariate_feature_scoring_function, weight_outputs=False):
    """
    Handle multivariate feature selection by running univariate on each output and averaging.
    :param weight_outputs: Use the mean + stddev of outputs to weight output importance
    """
    output_weights = Y.mean(axis=0) + Y.std(axis=0)

    scores = []
    for output_index in range(Y.shape[1]):
        output = Y[:, output_index]
        scores.append(univariate_feature_scoring_function(X, output))

    score_matrix = numpy.vstack(scores) # M outputs x N features

    if weight_outputs:
        # should be 1 x N
        return output_weights.dot(score_matrix)
    else:
        return score_matrix.mean(axis=0)
