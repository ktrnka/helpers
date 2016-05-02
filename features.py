from __future__ import unicode_literals
from __future__ import print_function

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


class TimeRange(object):
    """Thin wrapper to help deal with events that span a range of time"""
    def __init__(self, start, end):
        self.start = start
        self.end = end
