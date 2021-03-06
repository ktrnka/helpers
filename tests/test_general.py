import unittest

import numpy
import pandas

from helpers.general import add_temporal_noise
from ..features import TimeRange, get_event_series, add_roll, roll
from .. import general


class TimeShiftTests(unittest.TestCase):
    @staticmethod
    def _get_data(n=100):
        X = numpy.asarray(range(n))
        return numpy.vstack([X, X ** 2, X ** 3]).transpose()

    def test_errors(self):
        X = self._get_data()

        self.assertRaises(ValueError, general.prepare_time_matrix, X, 0)
        self.assertRaises(ValueError, general.prepare_time_matrix, X, -4)

    def test_identity(self):
        X = self._get_data()

        X_1 = general.prepare_time_matrix(X, 1)
        self.assertEqual(X.shape[0], X_1.shape[0])
        self.assertEqual(X.shape[1], X_1.shape[2])
        self.assertEqual(1, X_1.shape[1])
        self.assertTrue(numpy.array_equal(X, X_1.reshape(X.shape)))

    def test_simple(self):
        X = self._get_data(10)

        # basic tests - each row is x, x**2, x**3
        self.assertEqual(X[0, 1], 0)
        self.assertEqual(X[5, 1], 25)
        self.assertEqual(X[5, 2], 125)

        X_time = general.prepare_time_matrix(X, 5)

        self.assertSequenceEqual((X.shape[0], 5, X.shape[1]), X_time.shape)

        # the last index is the current value
        self.assertEqual(X_time[0, -1, 1], 0)
        self.assertEqual(X_time[5, -1, 1], 25)
        self.assertEqual(X_time[5, -1, 2], 125)

        # test shifted into past 1 step
        self.assertEqual(X_time[5, -2, 0], 4)
        self.assertEqual(X_time[5, -2, 1], 16)
        self.assertEqual(X_time[5, -2, 2], 64)

        self.assertEqual(X_time[5, -5, 0], 1)
        self.assertEqual(X_time[5, -5, 1], 1)
        self.assertEqual(X_time[5, -5, 2], 1)

        self.assertEqual(X_time[9, -5, 0], 5)
        self.assertEqual(X_time[9, -5, 1], 25)
        self.assertEqual(X_time[9, -5, 2], 125)


        # by default it wraps around
        self.assertEqual(X_time[0, -2, 0], 9)
        self.assertEqual(X_time[0, -2, 1], 81)
        self.assertEqual(X_time[0, -2, 2], 729)

    def test_no_rotation(self):
        X = self._get_data(10)
        X_time = general.prepare_time_matrix(X, 5, fill_value=-1)

        self.assertEqual(X_time[5, -5, 0], 1)
        self.assertEqual(X_time[5, -5, 1], 1)
        self.assertEqual(X_time[5, -5, 2], 1)

        self.assertEqual(X_time[0, -2, 0], -1)
        self.assertEqual(X_time[0, -2, 1], -1)
        self.assertEqual(X_time[0, -2, 2], -1)

        # just check the squares cause the fill val is negative
        self.assertEqual(X_time[2, -2, 1], 1)
        self.assertEqual(X_time[2, -3, 1], 0)
        self.assertEqual(X_time[2, -4, 1], -1)
        self.assertEqual(X_time[2, -5, 1], -1)

    def test_temporal_noise(self):
        ones = numpy.ones((10, 10))
        self.assertAlmostEqual(0, (ones - add_temporal_noise(ones)).sum())

        # random shouldn't be close
        r = numpy.random.rand(10, 10)
        self.assertNotAlmostEqual(0, (r - add_temporal_noise(r)).sum())

        # try with identical random features
        random_features = numpy.random.rand(1, 10)
        ident_rows = numpy.repeat(random_features, 20, axis=0)
        def_ident_rows = add_temporal_noise(ident_rows)
        self.assertAlmostEqual(0, (ident_rows - def_ident_rows).sum())

        # try with monotonic increasing
        monotonic = numpy.repeat(numpy.asarray(range(1, 1001)).reshape((1000, 1)), 20, axis=1)
        monotonic_deformed = add_temporal_noise(monotonic)

        self.assertLessEqual((abs(monotonic - monotonic_deformed) / monotonic).mean(), .05)


class TimeRangeTests(unittest.TestCase):
    def _make_time(self, start_time, duration_minutes=30):
        duration = pandas.Timedelta(minutes=duration_minutes)
        end_time = start_time + duration
        return TimeRange(start_time, end_time)

    def test_simple(self):
        """Test basic event-filling functionality"""
        hourly_index = pandas.DatetimeIndex(freq="1H", start=pandas.datetime(year=2016, month=4, day=1), periods=1000)

        dummy_events = [self._make_time(pandas.datetime(year=2016, month=4, day=1, hour=5, minute=50)), self._make_time(pandas.datetime(year=2016, month=4, day=1, hour=7, minute=20))]

        indicatored = get_event_series(hourly_index, dummy_events)
        self.assertEqual(1, indicatored.sum())

        minute_index = pandas.DatetimeIndex(freq="1Min", start=pandas.datetime(year=2016, month=4, day=1), periods=1000)
        indicatored = get_event_series(minute_index, dummy_events)
        self.assertEqual(60, indicatored.sum())


class FeatureTransformTests(unittest.TestCase):
    def test_lag_features(self):
        # make some monotonically increasing data
        datetime_index = pandas.date_range("1/1/2011", periods=1000, freq="M")
        data_series = pandas.Series(range(1000), index=datetime_index)

        data = pandas.DataFrame(index=datetime_index)
        data["my_feature"] = data_series

        add_roll(data, "my_feature", 60, "1h")
        add_roll(data, "my_feature", -60, "next1h")

        # make sure it's computing on the correct side of the index
        self.assertLess(data["my_feature"][60], data["my_feature_rolling_next1h"][60])
        self.assertGreater(data["my_feature"][60], data["my_feature_rolling_1h"][60])

        # test backfilling
        self.assertEqual(data["my_feature_rolling_1h"][50], data["my_feature_rolling_1h"][40])
        self.assertEqual(data["my_feature_rolling_next1h"][-50], data["my_feature_rolling_next1h"][-40])

        # test that they're just shifted versions of each other after the backfill spans
        self.assertEqual(data["my_feature_rolling_1h"][149], data["my_feature_rolling_next1h"][90])

    def test_functional_rolling(self):
        """Test the version of rolling helpers that doesn't have side effects"""
        datetime_index = pandas.date_range("1/1/2011", periods=1000, freq="M")
        data_series = pandas.Series(range(1000), index=datetime_index)

        data = pandas.DataFrame(index=datetime_index)
        data["my_feature"] = data_series

        # compute it just slightly differently
        data["my_feature_rolling_1h"] = roll(data["my_feature"], 60)
        data["my_feature_rolling_next1h"] = roll(data["my_feature"], -60)

        # try the first few tests
        self.assertLess(data["my_feature"][60], data["my_feature_rolling_next1h"][60])
        self.assertGreater(data["my_feature"][60], data["my_feature_rolling_1h"][60])

        # test that it'll work on whole dataframes not just series (cause the pandas interface for rolling is the same for both)
        df_rolled_60 = roll(data[["my_feature"]], 60)
        df_rolled_next60 = roll(data[["my_feature"]], -60)

        self.assertTrue(isinstance(df_rolled_60, pandas.DataFrame))
        self.assertLess(data["my_feature"][60], df_rolled_next60["my_feature"][60])
        self.assertGreater(data["my_feature"][60], df_rolled_60["my_feature"][60])


class DataSetTests(unittest.TestCase):
    def test_feature_selection(self):
        num_features = 10
        feature_names = pandas.Series(["feature_{}".format(i) for i in range(10)])

        dataset = general.DataSet(numpy.random.rand(100, num_features), numpy.random.rand(100, 2), None, feature_names, None, None)

        weights = numpy.random.rand(dataset.inputs.shape[1], 1).flatten()
        reduced = dataset.select_features(5, weights)

        self.assertEqual(5, reduced.inputs.shape[1])

        weights = numpy.random.rand(5, 1).flatten()
        self.assertRaises(ValueError, dataset.select_features, 5, weights)

    def test_argchecks(self):
        """Make sure that the DataSet is reasonably well formed on building it"""
        self.assertRaises(ValueError, general.DataSet, numpy.random.rand(100, 2), numpy.random.rand(50, 1), None, None, None, None)

