from __future__ import print_function
from __future__ import unicode_literals

import datetime
import inspect
import logging
import os.path
import random
import re
import time
from operator import itemgetter

import numpy


def number_string(number, singular_unit, plural_unit, format_string="{} {}"):
    return format_string.format(number, singular_unit if number == 1 else plural_unit)


class Timed(object):
    """Decorator for timing how long a function takes"""

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        start_time = time.time()
        retval = self.func(*args, **kwargs)
        elapsed = time.time() - start_time

        hours, seconds = divmod(elapsed, 60 * 60)
        minutes = seconds / 60.
        time_string = number_string(minutes, "minute", "minutes", format_string="{:.1f} {}")
        if hours:
            time_string = ", ".join((number_string(hours, "hour", "hours"), time_string))

        print("{} took {}".format(self.func.__name__, time_string))

        return retval


class DataSet(object):
    """Thin wrapper to bundle common data vars"""
    def __init__(self, inputs, outputs, splits, feature_names, target_names, output_index, split_map=None):
        """
        Group related vars into a single data set object that encapsulates the inputs, outputs, feature names, cross-validation
        splitting, and so on.
        :param inputs: Feature matrix, typically called X
        :param outputs: Output matrix, typically called y or Y
        :param splits: sklearn cross-validation iterator
        :param feature_names: For select_features to work this must be a pandas Series or numpy ndarray (something that supports using a list as index)
        :param target_names: Names for the output columns
        :param output_index: The index for the outputs. This can be important if you need to retain say time-series labels but the models only support input as ndarray
        :param split_map: Mapping of cross-validation names to cross-validation iterators. For supporting multiple CVs.
        :return:
        """
        if inputs.shape[0] != outputs.shape[0]:
            raise ValueError("input has {} rows, output has {} rows".format(inputs.shape[0], outputs.shape[0]))
        self.inputs = inputs
        self.outputs = outputs
        self.splits = splits
        self.feature_names = feature_names
        self.target_names = target_names
        self.output_index = output_index

        # mapping of labels to cross validation splits to support multi CV
        self.split_map = split_map

    def select_features(self, num_features, feature_scores, higher_is_better=True, verbose=0):
        if len(feature_scores) != len(self.feature_names):
            raise ValueError("feature_scores is size {} but there are {} features".format(len(feature_scores), self.inputs.shape[1]))

        pairs = enumerate(feature_scores)
        pairs = sorted(pairs, key=itemgetter(1), reverse=higher_is_better)

        if num_features < 1:
            num_features = int(num_features * self.inputs.shape[1])

        if verbose >= 1:
            print([(self.feature_names[i], score) for i, score in pairs])

        indexes = [p[0] for p in pairs]
        indexes = indexes[:num_features]

        return DataSet(self.inputs[:, indexes], self.outputs, self.splits, self.feature_names[indexes], self.target_names, self.output_index, split_map=self.split_map)

    def select_nonzero_features(self, feature_scores, verbose=0):
        selector = feature_scores != 0

        if verbose:
            print("Pruning {} zero-weight features from {}".format(len(feature_scores) - selector.sum(), len(feature_scores)))
        return DataSet(self.inputs[:, selector], self.outputs, self.splits, self.feature_names[selector], self.target_names, self.output_index)


def prepare_time_matrix(X, time_steps=5, fill_value=None):
    if time_steps < 1:
        raise ValueError("time_steps must be 1 or more")

    assert isinstance(X, numpy.ndarray)
    time_shifts = [X]
    time_shifts.extend(numpy.roll(X, t, axis=0) for t in range(1, time_steps))
    time_shifts = reversed(time_shifts)

    X_time = numpy.dstack(time_shifts)
    X_time = X_time.swapaxes(1, 2)

    if fill_value is not None:
        for t in range(time_steps):
            missing_steps = time_steps - t
            X_time[t, :missing_steps - 1, :] = fill_value

    return X_time


def add_temporal_noise(X, weight=0.05):
    assert isinstance(X, numpy.ndarray)

    X_noised = X.copy()

    for row in range(X_noised.shape[0] - 1):
        previous_weight = random.uniform(0., weight)
        next_weight = random.uniform(0., weight)

        # sum to 1: we're doing a weighted average
        total = 1. + previous_weight + next_weight
        ident = 1. / total
        previous_weight /= total
        next_weight /= total

        X_noised[row] = ident * X[row] + previous_weight * X[row - 1] + next_weight * X[row + 1]

    return X_noised



def _with_extra(filename, extra_info):
    base, ext = os.path.splitext(filename)
    return "".join([base, ".", extra_info, ext])


def with_num_features(filename, X):
    return _with_extra(filename, "{}_features".format(X.shape[1]))


def camel_to_snake(name):
    """From http://stackoverflow.com/a/1176023/1492373"""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def with_date(filename):
    return _with_extra(filename, datetime.datetime.now().strftime("%m_%d"))


def get_function_logger(num_calls_ago=1):
    _, file_name, _, function_name, _, _ = inspect.stack()[num_calls_ago]
    if file_name:
        file_name = os.path.basename(file_name)
    return logging.getLogger("{}:{}".format(file_name, function_name))


def get_class_logger(obj):
    return logging.getLogger(type(obj).__name__)
