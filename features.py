from __future__ import unicode_literals
from __future__ import print_function

from operator import itemgetter

import numpy
import pandas

"""
Helpers for derived features, particularly in Pandas.
"""


def add_lag_feature(dataframe, feature, time_steps, time_suffix, drop=False, data_type=None):
    """Derive a rolling mean with the specified time steps"""
    assert isinstance(dataframe, pandas.DataFrame)
    assert isinstance(dataframe.index, pandas.DatetimeIndex)

    name = feature + "_rolling_{}".format(time_suffix)
    dataframe[name] = dataframe[feature].rolling(window=time_steps).mean().fillna(method="backfill")

    if data_type:
        dataframe[name] = dataframe[name].astype(data_type)

    if drop:
        dataframe.drop([feature], axis=1, inplace=True)


def add_transformation_feature(dataframe, feature, transform, drop=False):
    """Derive a feature with a simple function like log, sqrt, etc"""
    assert isinstance(dataframe, pandas.DataFrame)
    new_name = feature + "_" + transform

    if transform == "log":
        transformed = numpy.log(dataframe[feature] + 1)
    elif transform == "square":
        transformed = numpy.square(dataframe[feature])
    elif transform == "sqrt":
        transformed = numpy.sqrt(dataframe[feature])
    elif transform == "gradient":
        transformed = numpy.gradient(dataframe[feature])
    else:
        raise ValueError("Unknown transform {} specified".format(transform))

    dataframe[new_name] = transformed

    if drop:
        dataframe.drop([feature], axis=1, inplace=True)


def get_event_series(datetime_index, event_ranges):
    """Create a boolean series showing when in the datetime_index we're in the time ranges in the event_ranges"""
    assert isinstance(datetime_index, pandas.DatetimeIndex)
    series = pandas.Series(data=0, index=datetime_index, dtype=numpy.int8)

    for event in event_ranges:
        assert isinstance(event, TimeRange)

        # Note: the side only matters for exact matches which are super rare in a datetime index
        closest_start = series.index.searchsorted(event.start, side="right")
        closest_end = series.index.searchsorted(event.end, side="right")
        series.loc[closest_start:closest_end] = 1

    return series


def find_best_features(dataset, model, scorer, n_jobs=1):
    import sklearn.cross_validation

    baseline = sklearn.cross_validation.cross_val_score(model, dataset.inputs, dataset.outputs, scoring=scorer, cv=dataset.splits, n_jobs=n_jobs).mean()

    # try deleting each feature
    loo_scores = [None for _ in dataset.feature_names]
    for i, name in enumerate(dataset.feature_names):
        included = [j for j in range(dataset.inputs.shape[1]) if j != i]

        reduced_inputs = dataset.inputs[:, included]
        loo_scores[i] = sklearn.cross_validation.cross_val_score(model, reduced_inputs, dataset.outputs, scoring=scorer, cv=dataset.splits, n_jobs=n_jobs).mean()

    # rank features by LOO scores
    ranked_features = sorted(enumerate(loo_scores), key=itemgetter(1), reverse=True)
    pruned_scores = [None for _ in ranked_features]
    pruned_scores[0] = baseline

    # assume that we dropping them in their LOO order is optimal (not generally true but it might work)
    prune_set = set()
    for i, _ in ranked_features[:-1]:
        prune_set.add(i)
        included = [j for j in range(dataset.inputs.shape[1]) if j not in prune_set]
        reduced_inputs = dataset.inputs[:, included]
        pruned_scores[len(prune_set)] = sklearn.cross_validation.cross_val_score(model, reduced_inputs, dataset.outputs, scoring=scorer, cv=dataset.splits, n_jobs=n_jobs).mean()

        print("Score with {} features: {}".format(len(included), pruned_scores[len(prune_set)]))


class TimeRange(object):
    """Thin wrapper to help deal with events that span a range of time"""
    def __init__(self, start, end):
        self.start = start
        self.end = end
