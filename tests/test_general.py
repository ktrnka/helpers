import unittest

import numpy
import pandas

from helpers.features import TimeRange, get_event_series
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
