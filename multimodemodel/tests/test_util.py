# flake8: noqa
"""Test utility functions."""
import pytest
import datetime

from multimodemodel.util import (
    npdatetime64_to_timestamp,
    timestamp_to_npdatetime64,
    str_to_date,
    average_npdatetime64,
)


@pytest.fixture(
    params=(
        ("2000-01-01", "%Y-%m-%d"),
        ("0001-01-01 10:00", "%Y-%m-%d %H:%M"),
        ("9999-12-31 10:00:00", "%Y-%m-%d %H:%M:%S"),
        ("9999-12-31 10:00:00.000000", "%Y-%m-%d %H:%M:%S.%f"),
    )
)
def date_str_frmt(request):
    return request.param[0], request.param[1]


@pytest.fixture
def date_str(date_str_frmt):
    return date_str_frmt[0]


@pytest.fixture
def dt64_tuple():
    return (
        tuple(
            str_to_date(d)
            for d in (
                "2000-01-01",
                "2000-01-02",
                "2000-01-03",
                "2000-01-04 00:00:00.0001",
            )
        ),
        str_to_date("2000-01-02 12:00:00.000025"),
    )


def test_time_conversion(date_str_frmt):
    date_str, frmt = date_str_frmt
    dt64 = str_to_date(date_str)
    dt = datetime.datetime.strptime(date_str, frmt)
    assert dt64 == dt


def test_timestamp_roundtripping(date_str):
    npdt64 = str_to_date(date_str)
    ts = npdatetime64_to_timestamp((npdt64,))[0]
    npdt64_roundtrip = timestamp_to_npdatetime64(ts)
    assert npdt64_roundtrip == npdt64


def test_average_dates(dt64_tuple):
    dt64s, oracle = dt64_tuple
    assert average_npdatetime64(dt64s) == oracle
