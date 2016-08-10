from __future__ import print_function
from __future__ import unicode_literals

import numpy
import pandas

"""
Helpers for derived features in Pandas.
"""


def add_roll(dataframe, feature_name, time_steps, time_suffix, drop=False, data_type=None, use_ewma=False, min_periods=None):
    """Add a rolling mean with the specified time steps"""
    assert isinstance(dataframe, pandas.DataFrame)
    assert isinstance(dataframe.index, pandas.DatetimeIndex)

    name = feature_name + "_rolling_{}".format(time_suffix)
    if use_ewma:
        name += "_ewma"

    if use_ewma:
        dataframe[name] = dataframe[feature_name].ewm(span=time_steps, min_periods=min_periods).mean().bfill()
    else:
        if time_steps < 0:
            dataframe[name] = dataframe[feature_name][::-1].rolling(window=-time_steps, min_periods=min_periods).mean().bfill()[::-1]
        else:
            dataframe[name] = dataframe[feature_name].rolling(window=time_steps, min_periods=min_periods).mean().bfill()

    if data_type:
        dataframe[name] = dataframe[name].astype(data_type)

    if drop:
        dataframe.drop([feature_name], axis=1, inplace=True)


def roll(series, num_steps, function="mean", min_periods=None):
    """Helper to roll a series forwards or backwards without inplace modification, can specify mean or sum"""
    run_backwards = num_steps < 0

    # if num steps is negative we want the rolling to happen into the future so we reverse the data, roll, then reverse again
    if run_backwards:
        num_steps = -num_steps
        series = series[::-1]

    # rolling into the past
    rolled = series.rolling(num_steps, min_periods=min_periods)

    # apply the function
    if function == "mean":
        rolled = rolled.mean()
    elif function == "sum":
        rolled = rolled.sum()
    else:
        raise ValueError("Unknown function {}".format(function))

    # cleanup
    rolled = rolled.bfill()

    if run_backwards:
        rolled = rolled[::-1]

    return rolled


def add_transform(dataframe, feature_name, transform, drop=False):
    """Add a feature with a simple function like log, sqrt, etc"""
    assert isinstance(dataframe, pandas.DataFrame)
    new_name = feature_name + "_" + transform

    if transform == "log":
        transformed = numpy.log(dataframe[feature_name] + 1)
    elif transform == "square":
        transformed = numpy.square(dataframe[feature_name])
    elif transform == "sqrt":
        transformed = numpy.sqrt(dataframe[feature_name])
    elif transform == "gradient":
        transformed = numpy.gradient(dataframe[feature_name])
    else:
        raise ValueError("Unknown transform {} specified".format(transform))

    dataframe[new_name] = transformed

    if drop:
        dataframe.drop([feature_name], axis=1, inplace=True)


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


class TimeRange(object):
    """Thin wrapper to help deal with events that span a range of time"""
    def __init__(self, start, end):
        self.start = start
        self.end = end


