# Contributing Guide

### Use pre-commit hooks

To enforce code style, we use [pre-commit](https://pre-commit.com/) hooks.
These hooks will be run by `git` before every commit and perform several checks and file manipulations.
To enable pre-commit, make sure to install all packages from the `requirements.txt` file and run
```shell
pre-commit install
```
within the project directory.
