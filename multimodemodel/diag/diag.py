"""Diagnostics to evaluate code performance."""

from typing import OrderedDict
from ..API_implementation import ParameterSplit, GridSplit
from ..grid import Grid


def lru_cache_info() -> OrderedDict:
    """Return map of method names to cache info."""
    cache_info = OrderedDict()
    cache_info["Grid.__eq__grid__"] = Grid.__eq__grid__.cache_info()
    cache_info["GridSplit.split"] = GridSplit.split.cache_info()
    cache_info["GridSplit.merge.__wrapped__"] = GridSplit.merge.__wrapped__.cache_info()
    cache_info["ParameterSplit.split"] = ParameterSplit.split.cache_info()
    cache_info[
        "ParameterSplit.merge.__wrapped__"
    ] = ParameterSplit.merge.__wrapped__.cache_info()
    return cache_info


def print_lru_cache_info():
    """Print aggregated lru cache info."""
    info = lru_cache_info()
    max_len = max(map(len, info.keys()))
    print("lru_cache Info:\n---------------")
    for m, i in info.items():
        print(f"{m:{max_len}}: {i}")
