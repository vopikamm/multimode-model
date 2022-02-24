"""Diagnostics to evaluate code performance."""

from typing import OrderedDict
from multimodemodel import Parameter, Grid


def lru_cache_info() -> OrderedDict:
    """Return map of method names to cache info."""
    cache_info = OrderedDict()
    cache_info["Grid.__eq__grid__"] = Grid.__eq__grid__.cache_info()
    cache_info["Grid.split"] = Grid.split.cache_info()
    cache_info["Grid.merge"] = Grid.merge.cache_info()
    cache_info["Parameter.split"] = Parameter.split.cache_info()
    cache_info["Parameter.merge"] = Parameter.merge.cache_info()
    return cache_info


def print_lru_cache_info():
    """Print aggregated lru cache info."""
    info = lru_cache_info()
    max_len = max(map(len, info.keys()))
    print("lru_cache Info:\n---------------")
    for m, i in info.items():
        print(f"{m:{max_len}}: {i}")
