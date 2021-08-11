# Contributing Guide

### Use pre-commit hooks

To enforce code style, we use [pre-commit](https://pre-commit.com/) hooks.
These hooks will be run by `git` before every commit and perform several checks and file manipulations.
To enable pre-commit, make sure to install all packages from the `requirements.txt` file and run
```shell
pre-commit install
```
within the project directory.

## Benchmark performance
For quantifying the effect of code changes on performance, a suite of benchmark test are available.
These are run by the [benchmark plugin](https://pytest-benchmark.readthedocs.io/en/latest/) of pytest.
Baseline performance metrics are produced **prior to any code change** by running

```bash
py.test --benchmark-save="baseline" benchmark
```

To compare performance changes while changing the code, you can run

```bash
py.test --benchmark-compare="*_baseline" benchmark
```
