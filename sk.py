from __future__ import print_function
from __future__ import unicode_literals

import collections
import itertools
import math
import numbers
from operator import itemgetter

import numpy
import pandas
import scipy.optimize
import scipy.stats
import sklearn
import sklearn.base
import sklearn.linear_model
import sklearn.metrics
import sklearn.utils.random
import json

from . import general


def _convert_scale(target_value, max_value):
    """If target_value is float, mult with max_value otherwise take it straight"""
    if target_value <= 1:
        return int(max_value * target_value)

    assert target_value <= max_value

    return target_value


def get_lr_importances(model):
    return numpy.abs(model.coef_).max(axis=0)


class SubspaceWrapper(sklearn.base.BaseEstimator):
    """Train the nested estimator with a random subspace"""
    def __init__(self, base_estimator=None, max_samples=1.0, max_features=1.0):
        # TODO: Add random seed
        self.base_estimator = base_estimator
        self.max_samples = max_samples
        self.max_features = max_features

        self.logger_ = general.get_class_logger(self)
        self.cols_ = None
        self.estimator_ = None

    def fit(self, X, Y, feature_probs=None):
        assert isinstance(X, numpy.ndarray)
        assert isinstance(Y, numpy.ndarray)

        rows = sklearn.utils.random.sample_without_replacement(X.shape[0], _convert_scale(self.max_samples, X.shape[0]))

        if feature_probs is not None:
            self.cols_ = numpy.random.choice(X.shape[1], _convert_scale(self.max_features, X.shape[1]), replace=False, p=feature_probs)
        else:
            self.cols_ = sklearn.utils.random.sample_without_replacement(X.shape[1], _convert_scale(self.max_features, X.shape[1]))

        self.logger_.debug("X.shape: %s", X.shape)
        self.logger_.debug("Y.shape: %s", Y.shape)
        self.logger_.debug("Selecting %d x %d subspace", len(rows), len(self.cols_))

        self.logger_.debug("Rows for %f: %s", self.max_samples, rows)
        self.logger_.debug("Cols for %f: %s", self.max_features, self.cols_)

        self.estimator_ = sklearn.base.clone(self.base_estimator).fit(X[rows][:, self.cols_], Y[rows])
        return self

    def predict(self, X):
        assert isinstance(X, numpy.ndarray)
        return self.estimator_.predict(X[:, self.cols_])

    def get_feature_weights(self, feature_weight_getter):
        return zip(self.cols_, feature_weight_getter(self.estimator_))


class MultivariateBaggingRegressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    """Bagging model that will use an underlying multivariate regression model unlike sklearn bagging"""
    def __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, feature_weight_getter=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.feature_weight_getter = feature_weight_getter

        self.logger = general.get_class_logger(self)
        self.estimators_ = None
        self.num_features_ = None

    def _get_estimator(self):
        return SubspaceWrapper(self.base_estimator, self.max_samples, self.max_features)

    def fit(self, X, Y):
        assert len(Y.shape) == 2
        self.num_features_ = X.shape[1]

        if not self.feature_weight_getter:
            self.estimators_ = [self._get_estimator().fit(X, Y) for _ in range(self.n_estimators)]
        else:
            sample1 = self.n_estimators / 3
            sample2 = self.n_estimators - sample1

            self.estimators_ = [self._get_estimator().fit(X, Y) for _ in range(sample1)]
            feature_probs = self.get_feature_weights(self.feature_weight_getter)
            self.estimators_.extend(self._get_estimator().fit(X, Y, feature_probs=feature_probs) for _ in range(sample2))

        return self

    def predict(self, X):
        result = numpy.dstack([estimator.predict(X) for estimator in self.estimators_])

        assert len(result.shape) == 3
        assert result.shape[0] == X.shape[0]

        result = result.mean(axis=2)
        assert len(result.shape) == 2

        return result

    def evaluate_features(self, X, Y):
        feature_scores = collections.defaultdict(list)
        estimator_scores = list()
        for estimator in self.estimators_:
            score = _rms_error(Y, estimator.predict(X))
            estimator_scores.append(score)
            for feature in estimator.cols_:
                feature_scores[feature].append(score)

        overall_mean = numpy.mean(estimator_scores)

        # max across estimators
        return {f: numpy.mean(scores) - overall_mean for f, scores in feature_scores.items()}

    def evaluate_features_cv(self, X, Y, splits):
        scores = collections.defaultdict(list)
        for train, test in splits:
            self.fit(X[train], Y[train])
            split_scores = self.evaluate_features(X[test], Y[test])

            for feature, score in split_scores.items():
                scores[feature].append(score)

        # mean across splits
        return {f: numpy.mean(scores) for f, scores in scores.items()}

    def get_feature_weights(self, feature_weight_getter):
        weights = [list() for _ in range(self.num_features_)]

        for estimator in self.estimators_:
            for col, weight in estimator.get_feature_weights(feature_weight_getter):
                weights[col].append(weight)

        overall = numpy.mean(list(itertools.chain(*weights)))

        weights = numpy.asarray([numpy.min(w) if w else overall for w in weights])
        weights /= numpy.sum(weights)

        return weights


class TimeSeriesRegressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__(self):
        self.params_ = None

    @staticmethod
    def get_time_offset(X, base_time):
        assert isinstance(X, pandas.DataFrame)
        assert isinstance(X.index, pandas.DatetimeIndex)

        x = (X.index - base_time).astype(numpy.int64) / 10 ** 6
        return x

    @staticmethod
    def _time_function(t, base, linear_amplitude, log_amplitude, periodic_amplitude, periodic_period, periodic_offset):
        y = base + linear_amplitude * t + log_amplitude * numpy.log(t) + periodic_amplitude * numpy.sin(periodic_offset + t / (2 * math.pi * periodic_period))
        return y

    @staticmethod
    def _simple_time_function(t, base, linear_amplitude, periodic_amplitude, periodic_offset, periodic_period):
        y = base + linear_amplitude * t + periodic_amplitude * numpy.sin(periodic_offset + t / (2 * math.pi * periodic_period))
        return y

    @staticmethod
    def _get_time_function_defaults(y):
        time_range = y[-1] - y[0]
        numeric_range = y.max() - y.min()

        return y[0], time_range, time_range, numeric_range, 687 * 24 * 60 * 60 * 1000., 1

    def fit(self, x, y):
        assert len(x.shape) == 1
        optimal_params, covariance = scipy.optimize.curve_fit(TimeSeriesRegressor._simple_time_function, x, y, xtol=0.09)
        self.params_ = optimal_params

        return self

    def predict(self, x):
        assert len(x.shape) == 1
        return numpy.asarray([self._simple_time_function(t, *self.params_) for t in x])


class MultivariateRegressionWrapper(sklearn.base.BaseEstimator):
    """
    Wrap a univariate regression model to support multivariate regression.
    Tweaked from http://stats.stackexchange.com/a/153892

    Can remove once sklearn 0.8 is out
    """

    def __init__(self, estimator):
        self.estimator = estimator
        self.estimators_ = None

    def fit(self, X, Y):
        if isinstance(Y, pandas.DataFrame):
            Y = Y.values

        assert len(Y.shape) == 2
        self.estimators_ = [sklearn.base.clone(self.estimator).fit(X, Y[:, i]) for i in range(Y.shape[1])]
        return self

    def predict(self, X):
        result = numpy.hstack([estimator.predict(X)[:, numpy.newaxis] for estimator in self.estimators_])

        assert result.shape[0] == X.shape[0]
        assert result.shape[1] == len(self.estimators_)

        return result

    def get_best_param_distributions(self):
        """Get distributions over grid search parameters for the best models on each output"""

        params = collections.defaultdict(list)
        for estimator in self.estimators_:
            for k, v in estimator.best_params_.items():
                params[k].append(v)

        return {k: numpy.asarray(v) for k, v in params.items()}

    def print_best_params(self):
        print("Best hyperparameters for grid search inside of multivariate regression")

        for name, dist in self.get_best_param_distributions().items():
            try:
                print("\t{}: {:.2f} +/- {:.2f}".format(name, dist.mean(), dist.std()))
            except TypeError:
                print("\t{}: {}".format(name, collections.Counter(dist).most_common(1)))

    def get_feature_importances(self, feature_names):
        feature_importances = collections.defaultdict(list)

        for estimator in self.estimators_:
            try:
                importances = estimator.feature_importances_
            except AttributeError:
                try:
                    importances = estimator.best_estimator_.feature_importances_
                except AttributeError:
                    raise ValueError("Unable to find feature_importances_")

            for feature_name, feature_score in zip(feature_names, importances):
                feature_importances[feature_name].append(feature_score)

        return {k: numpy.asarray(v) for k, v in feature_importances.items()}

    def print_feature_importances(self, feature_names):
        print("Feature importances")

        scores = self.get_feature_importances(feature_names)
        for name, dist in sorted(scores.items(), key=lambda pair: pair[1].mean(), reverse=True):
            print("\t{}: {:.3f} +/- {:.3f}".format(name, dist.mean(), dist.std()))

class JoinedMultivariateRegressionWrapper(sklearn.base.BaseEstimator):
    """
    Wrap a univariate regression model to support multivariate regression.
    This uses the method that Alex mentioned on the forums - replicate the
    input once for each output with an indicator for the output
    """

    def __init__(self, estimator):
        self.estimator = estimator

        self.estimator_ = None
        self.num_outputs_ = None

    @staticmethod
    def _transform_outputs(Y):
        augmented_outputs = []
        for output_col in range(Y.shape[1]):
            outputs = Y[:, output_col].reshape((Y.shape[0], 1))
            augmented_outputs.append(outputs)

        Y_stacked = numpy.vstack(augmented_outputs).ravel()

        assert(Y_stacked.shape[0] == Y.shape[0] * Y.shape[1])

        return Y_stacked

    def _untransform_outputs(self, X_transformed, Y_transformed):
        # start empty
        Y = numpy.zeros((X_transformed.shape[0] / self.num_outputs_, self.num_outputs_))

        # fill in each output; this code doesn't make assumptions about how the stacking worked
        # just that the last col is the target output index
        for output_col in range(self.num_outputs_):
            selector = X_transformed[:, -1].astype(int) == output_col
            Y[:, output_col] = Y_transformed[selector]

        return Y

    def _transform_inputs(self, X):
        augmented_inputs = []
        for output_col in range(self.num_outputs_):
            inputs = numpy.hstack([X, numpy.asarray([output_col for _ in range(X.shape[0])]).reshape((X.shape[0], 1))])
            augmented_inputs.append(inputs)

        X_stacked = numpy.vstack(augmented_inputs)
        assert(X_stacked.shape[0] == X.shape[0] * self.num_outputs_)
        assert(X_stacked.shape[1] == X.shape[1] + 1)

        return X_stacked

    def fit(self, X, Y):
        assert(len(Y.shape) == 2 and Y.shape[1] > 1)

        self.num_outputs_ = Y.shape[1]

        X = self._transform_inputs(X)
        Y = self._transform_outputs(Y)
        
        X, Y = self._rearrange(X, Y, self.num_outputs_)

        self.estimator_ = sklearn.base.clone(self.estimator).fit(X, Y)

        return self

    def predict(self, X):
        X_transformed = self._transform_inputs(X)
        Y_transformed = self.estimator_.predict(X_transformed)
        result = self._untransform_outputs(X_transformed, Y_transformed)

        assert result.shape[0] == X.shape[0]
        assert result.shape[1] == self.num_outputs_

        return result

    @staticmethod
    def _rearrange(X, Y, num_outputs):
        """Rearrange the data so that it's all interleaved for any nested cross-validation"""
        base_rows = X.shape[0] / num_outputs
        row_order = []
        for base_index in range(base_rows):
            row_order.extend([j * base_rows + base_index for j in range(num_outputs)])
        return X[row_order], Y[row_order]


def print_tuning_scores(tuned_estimator, reverse=True):
    """Show the cross-validation scores and hyperparamters from a grid or random search"""
    for test in sorted(tuned_estimator.grid_scores_, key=itemgetter(1), reverse=reverse):
        scores = test.cv_validation_scores

        print("Validation score {:.4f} +/- {:.4f}, Hyperparams {}".format(scores.mean(),
                                                                          scores.std(),
                                                                          test.parameters))


def print_feature_importances(columns, classifier):
    """Show feature importances for a classifier that supports them like random forest or gradient boosting"""
    paired_features = zip(columns, classifier.feature_importances_)
    field_width = str(max(len(c) for c in columns))
    format_string = "\t{:" + field_width + "s}: {}"
    print("Feature importances")
    for feature_name, importance in sorted(paired_features, key=itemgetter(1), reverse=True):
        print(format_string.format(feature_name, importance))


class RandomizedSearchCV(sklearn.grid_search.RandomizedSearchCV):
    """Wrapper for sklearn RandomizedSearchCV that can run correlation analysis on hyperparameters"""

    def print_tuning_scores(self, reverse=True):
        for test in sorted(self.grid_scores_, key=itemgetter(1), reverse=reverse):
            scores = test.cv_validation_scores
            print("Validation score {:.4f} +/- {:.4f}, Hyperparams {}".format(scores.mean(),
                                                                              scores.std(),
                                                                              test.parameters))

        print("Hyperparameter correlations with evaluation metric")
        for param, (stat_name, stat, pval, extras) in self.correlate_hyperparameters().items():
            print("\t{}: {} = {:.4f}, p = {:.4f}".format(param, stat_name, stat, pval))

            if extras:
                for param_val, score in extras.most_common():
                    print("\t\t{}: {:.4f}".format(param_val, score))

    def correlate_hyperparameters(self):
        param_scores = self._get_independent_scores()

        param_correlations = dict()
        for param_name, points in param_scores.items():
            if all(isinstance(x, numbers.Number) for x, _ in points):
                # numeric params path: use Pearson
                points = numpy.asarray(points)
                assert points.shape[1] == 2

                pearson_r, pearson_p = scipy.stats.pearsonr(points[:, 0], points[:, 1])
                param_correlations[param_name] = ("Pearson r", pearson_r, pearson_p, None)
            else:
                # non-numeric path, run anova or something
                param_vals = collections.defaultdict(list)
                for param_val, score in points:
                    param_vals[param_val].append(score)

                anova_f, anova_p = scipy.stats.f_oneway(*[numpy.asarray(v) for v in param_vals.values()])

                extra = None
                if anova_p <= 0.05:
                    extra = collections.Counter({p: numpy.mean(vals) for p, vals in param_vals.items()})
                param_correlations[param_name] = ("Anova f", anova_f, anova_p, extra)

        return param_correlations

    def _get_independent_scores(self):
        param_scores = collections.defaultdict(list)
        param_counts = collections.defaultdict(collections.Counter)
        for test in self.grid_scores_:
            scores = test.cv_validation_scores

            for name, value in test.parameters.items():
                param_counts[name][value] += 1
                param_scores[name].append((value, scores.mean()))

        # remove parameter values that don't vary
        for param, value_distribution in param_counts.items():
            if len(value_distribution) == 1 and param in param_scores:
                del param_scores[param]

        return param_scores

    @staticmethod
    def uniform(start, end):
        """Helper to make a continuous or discrete uniform distribution depending on the input types"""
        if start > end:
            start, end = end, start
        if all(isinstance(x, int) for x in [start, end]):
            return scipy.stats.randint(start, end)
        else:
            return scipy.stats.uniform(start, end - start)

    @staticmethod
    def exponential(start, end, num_samples=100):
        """Helper to make a log-linear distribution"""
        if start > end:
            start, end = end, start
        return numpy.exp(numpy.linspace(math.log(start), math.log(end), num=num_samples))


class LinearRegressionWrapper(sklearn.linear_model.LinearRegression):
    """Wrapper for LinearRegression that's compatible with GradientBoostingClassifier sample_weights"""

    def fit(self, X, y, sample_weight=None):
        super(LinearRegressionWrapper, self).fit(X, y)

    def predict(self, X):
        return super(LinearRegressionWrapper, self).predict(X)[:, numpy.newaxis]


class ClippedRobustScaler(sklearn.preprocessing.RobustScaler):
    """Tweak on RobustScaler to clip values to -2 to 2 after reducing to IQR"""
    def __init__(self, clip_value=2, with_centering=True, with_scaling=True, copy=True):
        super(ClippedRobustScaler, self).__init__(with_centering=with_centering, with_scaling=with_scaling, copy=copy)
        self.clip_value = clip_value

    def transform(self, X, y=None):
        X = super(ClippedRobustScaler, self).transform(X, y)

        # try to ensure that -1 to 1 is a nice linear range and squash a bit beyond that
        X[X > self.clip_value] = self.clip_value
        X[X < -self.clip_value] = -self.clip_value

        return X


class TimeCV(object):
    """
    Cross-validation wrapper for time-series prediction, i.e., test only on extrapolations into the future.
    Assumes that the data is sorted chronologically.

    There doesn't seem to be an appropriate sklearn superclass because sklearn CV iterators
    all assume that you train on everything that isn't flagged for testing.
    """
    def __init__(self, num_rows, num_splits, min_training=0.5, test_splits=1, mirror=False, gap=0, balanced_tests=True):
        self.num_rows = int(num_rows)
        self.num_splits = int(num_splits)
        self.test_split_buckets = test_splits
        self.min_training = min_training

        self.mirror = mirror
        self.gap = gap
        self.balanced_tests = balanced_tests

        self.n = len(list(self.__iter__()))

    def __iter__(self):
        per_bin = self.num_rows / float(self.num_splits)

        for s in range(1, self.num_splits):
            train_end = int(per_bin * s)

            test_start = train_end + int(self.gap * per_bin)
            test_end = test_start + int(per_bin * self.test_split_buckets)

            if not self.balanced_tests:
                test_end = min(test_end, self.num_rows)

                # sometimes one leftover due to rounding error
                if test_end - test_start <= 1:
                    continue

            # only return uniform size tests
            if self.balanced_tests and test_end > self.num_rows:
                continue

            train_index = numpy.asarray(range(0, train_end), dtype=numpy.int32)
            test_index = numpy.asarray(range(test_start, test_end), dtype=numpy.int32)

            # skip any without enough data
            if train_end >= int(self.min_training * self.num_rows):
                yield list(train_index), list(test_index)

                if self.mirror:
                    yield list(self.num_rows - train_index - 1), list(self.num_rows - test_index - 1)

    def __len__(self):
        return self.n

class WraparoundTimeCV(object):
    def __init__(self, num_rows, num_splits, training_num_splits):
        self.num_rows = int(num_rows)
        self.num_splits = int(num_splits)
        self.training_num_splits = training_num_splits

        assert training_num_splits < num_splits

        self.n = len(list(self.__iter__()))

    def __iter__(self):
        per_bin = self.num_rows / float(self.num_splits)

        for test_bin in range(self.num_splits):
            test_start = int(per_bin * test_bin)
            test_end = int(per_bin * (test_bin + 1))

            training_first_bin = test_bin - self.training_num_splits

            train_index = []

            # tail of the training data
            if training_first_bin < 0:
                train_index.extend(range(int((training_first_bin % self.num_splits) * per_bin), self.num_rows))
                training_first_bin = 0
            # head of the training data
            train_index.extend(range(int(training_first_bin * per_bin), test_start))

            test_index = range(test_start, test_end)

            yield list(train_index), list(test_index)


    def __len__(self):
        return self.n

def _rms_error(y_true, y_pred):
    return float(numpy.square(y_true - y_pred).mean().mean() ** 0.5)


rms_error = sklearn.metrics.make_scorer(_rms_error, greater_is_better=False)


class OutputTransformation(sklearn.base.BaseEstimator):
    """
    Wraps an estimator such that the outputs are transformed by transformer before fitting and inverse transformed
    on predicting. This can be used with StandardScaler to remove the scale of outputs or can be used to augment the
    outputs.
    """

    def __init__(self, estimator, transformer):
        self.estimator = estimator
        self.transformer = transformer

        self.estimator_ = None
        self.transformer_ = None

    def fit(self, X, Y):
        self.transformer_ = sklearn.base.clone(self.transformer).fit(Y)
        self.estimator_ = sklearn.base.clone(self.estimator).fit(X, self.transformer_.transform(Y))
        return self

    def predict(self, X):
        return self.transformer_.inverse_transform(self.estimator_.predict(X))


class QuickTransform(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    def __init__(self, transform_function, inverse_function):
        self.transform_function = transform_function
        self.inverse_function = inverse_function

    def fit(self, data):
        return self

    def transform(self, data):
        return self.transform_function(data)

    def inverse_transform(self, transformed_data):
        return self.inverse_function(transformed_data)

    @staticmethod
    def make_non_negative():
        return QuickTransform(lambda Y: Y, lambda Y: numpy.maximum(Y, 0))

    @staticmethod
    def make_append_mean():
        return QuickTransform(lambda Y: numpy.hstack([Y, Y.mean(axis=1).reshape(-1, 1)]), lambda Y: Y[:, :-1])

    @staticmethod
    def make_append_rolling(window=7):
        def transform(Y):
            rolled = pandas.DataFrame(Y).rolling(window=window).mean().fillna(method="bfill")
            return numpy.hstack([Y, rolled.values * 0.25])

        def invert(Y):
            return Y[:, :Y.shape[1] / 2]

        return QuickTransform(transform, invert)

    @staticmethod
    def make_ewma_outputs(num_spans=3):
        return QuickTransform(lambda Y: Y, lambda Y: ewma_all(Y, num_spans))


def ewma_all(X, num_spans=3):
    df = pandas.DataFrame(X)
    forward = df.ewm(span=num_spans).mean()
    backward = df[::-1].ewm(span=num_spans).mean()[::-1]

    means = numpy.dstack([forward.values, backward.values]).mean(axis=2)
    assert(means.shape == X.shape)

    return means


def centered_ewma(series, num_spans):
    forward = series.ewm(span=num_spans).mean()
    backward = series[::-1].ewm(span=num_spans).mean()[::-1]

    means = numpy.vstack([forward.values, backward.values]).mean(axis=0)

    return pandas.Series(index=forward.index, data=means, name=series.name)


class OutputClippedTransform(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def fit(self, outputs):
        return self

    def transform(self, outputs):
        return outputs

    def inverse_transform(self, outputs):
        return numpy.minimum(numpy.maximum(outputs, self.min), self.max)

    @staticmethod
    def from_data(outputs):
        return OutputClippedTransform(outputs.min(axis=0), outputs.max(axis=0))

    @staticmethod
    def load(json_in):
        json_obj = json.load(json_in)

        self = OutputClippedTransform(numpy.asarray(json_obj["min_"]), numpy.asarray(json_obj["max_"]))

        return self

    def save(self, file):
        json_obj = {"min_": list(self.min), "max_": list(self.max)}
        json.dump(json_obj, file, indent=3)


def get_model_name(model, format="{}({})", remove={"Regressor", "Regression", "Classifier"}):
    """Get a nice string for a sklearn model with nested sklearn models"""
    name = type(model).__name__

    if remove:
        for substr in remove:
            name = name.replace(substr, "")

    try:
        nested_name = get_model_name(model.estimator, format)
        return format.format(name, nested_name)
    except AttributeError:
        try:
            nested_name = get_model_name(model.base_estimator, format)
            return format.format(name, nested_name)
        except AttributeError:
            try:
                nested_name = get_model_name(model._final_estimator, format)
                return format.format(name, nested_name)
            except AttributeError:
                return name


def with_model_name(filename, model, snake_case=False):
    model_name = get_model_name(model, format="{}_{}")
    if snake_case:
        model_name = general.camel_to_snake(model_name)
    return general._with_extra(filename, model_name)


def predictions_in_training_range(Y_train, Y_pred):
    """Fraction of predictions in the training data range (for testing regression interpolation vs extrapolation)"""
    Y_min = Y_train.min(axis=0)
    Y_max = Y_train.max(axis=0)

    in_range = (Y_pred <= Y_max[numpy.newaxis, :]) & (Y_pred >= Y_min[numpy.newaxis, :])

    return in_range.mean()


class AverageClonedRegressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__(self, base_estimator, num_clones=2):
        self.base_estimator = base_estimator
        self.num_clones = num_clones
        self.estimators_ = None

    def fit(self, X, Y):
        self.estimators_ = [sklearn.base.clone(self.base_estimator).fit(X, Y) for _ in range(self.num_clones)]
        return self

    def predict(self, X):
        predictions = numpy.dstack([e.predict(X) for e in self.estimators_])
        return predictions.mean(axis=2)


class DeltaSumRegressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__(self, base_estimator, num_rolls=2):
        self.base_estimator = base_estimator
        self.num_rolls = num_rolls

        self.value_estimator_ = None
        self.delta_estimator_ = None

    def fit(self, X, y):
        self.value_estimator_ = sklearn.base.clone(self.base_estimator).fit(X, y)

        # create deltas
        delta_y = (y - numpy.roll(y, 1, axis=0))[1:, :]
        slice_X = X[1:, :]
        self.delta_estimator_ = sklearn.base.clone(self.base_estimator).fit(slice_X, delta_y)

        return self

    def predict(self, X):
        y_pred = self.value_estimator_.predict(X)
        dy_pred = self.delta_estimator_.predict(X)

        predictions = [y_pred]

        # previous value plus dy
        for roll_window in range(1, self.num_rolls + 1):
            y_pred_step = numpy.roll(y_pred, roll_window, axis=0)
            for i in range(roll_window):
                y_pred_step += numpy.roll(dy_pred, i, axis=0)

                y_pred_step[:roll_window, :] = y_pred[:roll_window, :]

            percent_diff = (numpy.abs(y_pred - y_pred_step) / y_pred).mean() * 100
            print("Prediction difference from {}-step delta estimation: {:.1f}%".format(roll_window, percent_diff))

            predictions.append(y_pred_step)

        merged_predictions = numpy.mean(predictions, axis=0)

        assert merged_predictions.shape == y_pred.shape

        return merged_predictions


class WeightedEnsembleRegressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__(self, estimators, weights=None):
        self.estimators = estimators
        self.weights = weights

        # check that the weights are reasonable
        if weights:
            if len(weights) != len(estimators):
                raise ValueError("{} estimators but {} weights".format(len(estimators), len(weights)))

            if abs(sum(weights) - 1) > 0.1:
                raise ValueError("Weights should sum to 1, weights={}, sum={}".format(weights, sum(weights)))

        self.estimators_ = None

    def fit(self, X, Y):
        self.estimators_ = [sklearn.base.clone(estimator_template).fit(X, Y) for estimator_template in self.estimators]
        return self

    def predict(self, X):
        predictions = numpy.dstack([estimator.predict(X) for estimator in self.estimators_])
        return numpy.average(predictions, axis=2, weights=self.weights)


class StackedEnsembleRegressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__(self, estimators, arbiter, val=0.1):
        self.estimators = estimators
        self.arbiter = arbiter
        self.val = val

        self.estimators_ = None
        self.arbiter_ = None

    def fit(self, X, Y):
        # fit estimators, assuming that the last val amount of data is held out for early stopping and the like
        self.estimators_ = [sklearn.base.clone(estimator_template).fit(X, Y) for estimator_template in self.estimators]

        val_start = int(X.shape[0] * (1 - self.val))
        val_X = X[val_start:]
        val_Y = Y[val_start:]
        val_predictions = numpy.hstack([estimator.predict(val_X) for estimator in self.estimators_])

        self.arbiter_ = sklearn.base.clone(self.arbiter).fit(val_predictions, val_Y)
        return self

    def predict(self, X):
        predictions = numpy.hstack([estimator.predict(X) for estimator in self.estimators_])
        arbiter_predictions = self.arbiter_.predict(predictions)
        return arbiter_predictions


class ValidationWrapper(sklearn.base.BaseEstimator):
    def __init__(self, base_estimator, val=0.1):
        self.base_estimator = base_estimator
        self.val = val

        self.estimator_ = None

    def fit(self, X, Y):
        val_start = int((1 - self.val) * X.shape[0])
        self.estimator_ = sklearn.base.clone(self.base_estimator).fit(X[:val_start], Y[:val_start])
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


def with_val(model, val=0.1):
    return ValidationWrapper(model, val)


