"""Utility functions."""

from typing import Iterable
import numpy as np

# See https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-units
dt64_units = "us"
factor_relative_to_seconds = 1_000_000

dt64_dtype = f"datetime64[{dt64_units}]"


def str_to_date(date: str) -> np.datetime64:
    """Convert string to numpy.datetime64."""
    return np.datetime64(date, dt64_units)


def npdatetime64_to_timestamp(dates: Iterable[np.datetime64]) -> np.ndarray:
    """Convert sequence of numpy datetime64 to array of float timestamps."""
    return np.array(dates).astype(float)


def timestamp_to_npdatetime64(ts: float):
    """Convert float timestamp to numpy datetime64.

    Note that the timestamp will be rounded, hence there might be a loss of precision.
    Change `dt64_units` to increase precision.
    """
    return np.datetime64(round(ts), dt64_units)


def average_npdatetime64(dates: Iterable[np.datetime64]) -> np.datetime64:
    """Average numpy datetime64 objects.

    Precision is defined by `dt64_units`.
    """
    timestamps = npdatetime64_to_timestamp(dates)
    mean_timestamp = timestamps.mean()
    return timestamp_to_npdatetime64(mean_timestamp)


def add_time(date: np.datetime64, time: float) -> np.datetime64:
    """Add time to a date time object.

    Note that time will be converted to `dt64_units` and rounded, hence there might
    be a loss of precision. Change `dt64_units` to increase precision.
    """
    td = np.timedelta64(round(time * factor_relative_to_seconds), dt64_units)
    return np.datetime64(date + td, dt64_units)
