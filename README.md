[![pipeline status](https://git.geomar.de/mcgroup/multimode-model/badges/master/pipeline.svg)](https://git.geomar.de/mcgroup/multimode-model/commits/develop)
[![coverage report](https://git.geomar.de/mcgroup/multimode-model/badges/master/coverage.svg)](https://git.geomar.de/mcgroup/multimode-model/commits/develop)

# Multimode Model

Implementation of a non-linear ocean model which expresses the state as a projection onto vertical normal modes. Hence, the vertical coordinate is quasi-continuous but the vertical resolution is limited by the truncation in vertical mode space.


## Contribution Guide

### Use pre-commit hooks

To enforce code style, we use [pre-commit](https://pre-commit.com/) hooks.
These hooks will be run by `git` before every commit and perform several checks and file manipulations.
To enable pre-commit, make sure to install all packages from the `requirements.txt` file and run
```shell
pre-commit install
```
within the project directory.
