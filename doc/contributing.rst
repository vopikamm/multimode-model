.. _contributing:

******************************
Contributing to multimodemodel
******************************


.. note::

  Large parts of this document came from the `Xarray Contributing
  Guide <http://xarray.pydata.org/en/stable/contributing.html>`_.

Where to start?
===============

All contributions, bug reports, bug fixes, documentation improvements,
enhancements, and ideas are welcome.

If you are brand new to *multimodemodel* or open-source development, we recommend going
through the `Gitlab "issues" tab <https://git.geomar.de/mcgroup/multimode-model/-/issues>`_
to find issues that interest you. There are a number of issues listed under
`Documentation <https://git.geomar.de/mcgroup/multimode-model/-/issues?scope=all&state=opened&label_name[]=documentation>`_
where you could start out. Once you've found an interesting issue, you can
return here to get your development environment setup.


.. _contributing.bug_reports:

Bug reports and enhancement requests
====================================

Bug reports are an important part of making *multimodemodel* more stable. Having a complete bug
report will allow others to reproduce the bug and provide insight into fixing. See
`this stackoverflow article <https://stackoverflow.com/help/mcve>`_ for tips on
writing a good bug report.

Trying out the bug-producing code on the *master* branch is often a worthwhile exercise
to confirm that the bug still exists. It is also worth searching existing bug reports and
pull requests to see if the issue has already been reported and/or fixed.

Bug reports must:

#. Include a short, self-contained Python snippet reproducing the problem.
   You can format the code nicely by using `Gitlab Flavored Markdown
   <https://git.geomar.de/help/user/markdown>`_::

      ```python
      import multimodemodel as mmm
      par = mmm.Parameters(...)

      ...
      ```

#. Include the full version string of *multimodemodel* and its dependencies. You can use the
   built in function::

      ```python
      import multimodemodel as mmm
      mmm.show_versions()

      ...
      ```

#. Explain why the current behavior is wrong/not desired and what you expect instead.

The issue will then show up to the *multimodemodel* community and be open to comments/ideas
from others.

Working with the code
=====================

Now that you have an issue you want to fix, enhancement to add, or documentation
to improve, you need to learn how to work with Gitlab and the *multimodemodel* code base.

Version control and Git
-----------------------

To the new user, working with Git is one of the more daunting aspects of contributing
to *multimodemodel*.  It can very quickly become overwhelming, but sticking to the guidelines
below will help keep the process straightforward and mostly trouble free.  As always,
if you are having difficulties please feel free to ask for help.

The code is hosted on `Gitlab at GEOMAR <https://git.geomar.de/mcgroup/multimode-model>`_.
We use `Git <http://git-scm.com/>`_ for
version control to allow many people to work together on the project.

Some great resources for learning Git:

* the `GitHub help pages <http://help.github.com/>`_.
* the `NumPy's documentation <http://docs.scipy.org/doc/numpy/dev/index.html>`_.
* Matthew Brett's `Pydagogue <http://matthew-brett.github.io/pydagogue/>`_.

Getting started with Git
~~~~~~~~~~~~~~~~~~~~~~~~

`GitHub has instructions <http://help.github.com/set-up-git-redirect>`__ for installing git,
setting up your SSH key, and configuring git.  All these steps need to be completed before
you can work seamlessly between your local repository and GitHub.

Creating a branch
~~~~~~~~~~~~~~~~~

You want your ``master`` branch to reflect only production-ready code, so create a
feature branch before making your changes. For example

.. code-block:: sh

    git branch shiny-new-feature
    git checkout shiny-new-feature

The above can be simplified to

.. code-block:: sh

    git checkout -b shiny-new-feature

This changes your working directory to the shiny-new-feature branch.  Keep any
changes in this branch specific to one bug or feature so it is clear
what the branch brings to *multimodemodel*. You can have many "shiny-new-features"
and switch in between them using the ``git checkout`` command.

To update this branch, you need to retrieve the changes from the ``master`` branch

.. code-block:: sh

    git fetch upstream
    git merge upstream/master

This will combine your commits with the latest *multimodemodel* git ``master``.  If this
leads to merge conflicts, you must resolve these before submitting your pull
request.  If you have uncommitted changes, you will need to ``git stash`` them
prior to updating.  This will effectively store your changes, which can be
reapplied after updating.

.. _contributing.dev_env:

Creating a development environment
----------------------------------

To test out code changes, you'll need to build *multimodemodel* from source, which
requires a Python environment. If you're making documentation changes, you can
skip to :ref:`contributing.documentation` but you won't be able to build the
documentation locally before pushing your changes.

.. _contributing.dev_container:

Using the devcontainer with VSCode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are using `VS Code <https://code.visualstudio.com/>`_ and have `docker <https://www.docker.com/>`_
installed, you can use the predefined development container instead of creating
your own environment. See
`VS Codes documentation <https://code.visualstudio.com/docs/remote/containers>`_
for details on working with development containers.

.. _contributiong.dev_python:

Creating a Python Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before starting any development, you'll need to create an isolated multimodemodel
development environment:

- Install either `Anaconda <https://www.anaconda.com/download/>`_ or `miniconda
  <https://conda.io/miniconda.html>`_
- Make sure your conda is up to date (``conda update conda``)
- Make sure that you have :ref:`cloned the repository <contributing.forking>`
- ``cd`` to the *multimodemodel* source directory

We'll now kick off a two-step process:

1. Install the build dependencies
2. Build and install multimodemodel

.. code-block:: sh

   # Create and activate the build environment
   conda create -c conda-forge -n multimodemodel-tests python=3.8 pip

   conda activate multimodemodel-tests

   # Build and install multimodemodel
   pip install -e .

At this point you should be able to import *multimodemodel* from your locally
built version:

.. code-block:: sh

   $ python  # start an interpreter
   >>> import multimodemodel
   >>> multimodemodel.__version__
   '0.10.0+dev46.g015daca'

This will create the new environment, and not touch any of your existing environments,
nor any existing Python installation.

To view your environments

.. code-block:: sh

      conda info -e

To return to your root environment

.. code-block:: sh

      conda deactivate

See the full `conda docs <http://conda.pydata.org/docs>`_.


.. _contributing.documentation:

Contributing to the documentation
=================================

If you're not the developer type, contributing to the documentation is still of
huge value. You don't even have to be an expert on *multimodemodel* to do so! In fact,
there are sections of the docs that are worse off after being written by
experts. If something in the docs doesn't make sense to you, updating the
relevant section after you figure it out is a great way to ensure it will help
the next person.

.. contents:: Documentation:
   :local:


About the *multimodemodel* documentation
----------------------------------------

The documentation is written in **reStructuredText**, which is almost like writing
in plain English, and built using `Sphinx <http://sphinx-doc.org/>`__. The
Sphinx Documentation has an excellent `introduction to reST
<http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`__. Review the Sphinx docs to perform more
complex changes to the documentation as well.

Some other important things to know about the docs:

- The *multimodemodel* documentation consists of two parts: the docstrings in the code
  itself and the docs in this folder ``multimodemodel/doc/``.

  The docstrings are meant to provide a clear explanation of the usage of the
  individual functions, while the documentation in this folder consists of
  tutorial-like overviews per topic together with some other information
  (what's new, installation, etc).

- The docstrings follow the **NumPy Docstring Standard**, which is used widely
  in the Scientific Python community. This standard specifies the format of
  the different sections of the docstring. See `this document
  <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_
  for a detailed explanation, or look at some of the existing functions to
  extend it in a similar manner.

- The tutorials make heavy use of the `ipython directive
  <http://matplotlib.org/sampledoc/ipython_directive.html>`_ sphinx extension.
  This directive lets you put code in the documentation which will be run
  during the doc build. For example:

  .. code:: rst

      .. ipython:: python

          x = 2
          x ** 3

  will be rendered as::

      In [1]: x = 2

      In [2]: x ** 3
      Out[2]: 8

  Almost all code examples in the docs are run (and the output saved) during the
  doc build. This approach means that code examples will always be up to date,
  but it does make building the docs a bit more complex.

- Our API documentation in ``doc/api.rst`` houses the auto-generated
  documentation from the docstrings. For classes, there are a few subtleties
  around controlling which methods and attributes have pages auto-generated.

  Every method should be included in a ``toctree`` in ``api.rst``, else Sphinx
  will emit a warning.


How to build the *multimodemodel* documentation
-----------------------------------------------

Requirements
~~~~~~~~~~~~
Make sure to follow the instructions on :ref:`creating a development environment above <contributing.dev_env>`, but
to build the docs you need additional dependencies.

.. code-block:: sh

    pip install -e .[docs]

Building the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Navigate to your local ``doc/`` directory in the console and run

.. code-block:: sh

    make html

Then you can find the HTML output in the folder ``multimodemodel/doc/_build/html/``.

The first time you build the docs, it will take quite a while because it has to run
all the code examples and build all the generated docstring pages. In subsequent
evocations, Sphinx will try to only build the pages that have been modified.

If you want to do a full clean build, do::

    make clean
    make html

.. _contributing.code:

Contributing to the code base
=============================

.. contents:: Code Base:
   :local:

Code standards
--------------

Writing good code is not just about what you write. It is also about *how* you
write it. During :ref:`Continuous Integration <contributing.ci>` testing, several
tools will be run to check your code for stylistic errors.
Generating any warnings will cause the test to fail.
Thus, good style is a requirement for submitting code to *multimodemodel*.

In addition, it is important that we
do not make sudden changes to the code that could have the potential to break
a lot of user code as a result, that is, we need it to be as *backwards compatible*
as possible to avoid mass breakages.

Code Formatting
~~~~~~~~~~~~~~~

multimodemodel uses several tools to ensure a consistent code format throughout the project:

- `Black <https://black.readthedocs.io/en/stable/>`_ for standardized
  code formatting
- `Flake8 <http://flake8.pycqa.org/en/latest/>`_ for general code quality

.. - `blackdoc <https://blackdoc.readthedocs.io/en/stable/>`_ for
..   standardized code formatting in documentation
.. - `isort <https://github.com/timothycrosley/isort>`_ for standardized order in imports.
..   See also `flake8-isort <https://github.com/gforcada/flake8-isort>`_.
.. - `mypy <http://mypy-lang.org/>`_ for static type checking on `type hints
..   <https://docs.python.org/3/library/typing.html>`_

We highly recommend that you setup `pre-commit hooks <https://pre-commit.com/>`_
to automatically run all the above tools every time you make a git commit. This
can be done by running

.. code-block:: sh

    pre-commit install

from the root of the multimodemodel repository. You can skip the pre-commit checks
with ``git commit --no-verify``.


Backwards Compatibility
~~~~~~~~~~~~~~~~~~~~~~~

Please try to maintain backwards compatibility. If you think breakage is
required, clearly state why as part of the pull request.

Be especially careful when changing function and method signatures, because any change
may require a deprecation warning. For example, if your pull request means that the
argument ``old_arg`` to ``func`` is no longer valid, instead of simply raising an error if
a user passes ``old_arg``, we would instead catch it:

.. code-block:: python

    def func(new_arg, old_arg=None):
        if old_arg is not None:
            from warnings import warn

            warn(
                "`old_arg` has been deprecated, and in the future will raise an error."
                "Please use `new_arg` from now on.",
                DeprecationWarning,
            )

            # Still do what the user intended here

This temporary check would then be removed in a subsequent version of multimodemodel.
This process of first warning users before actually breaking their code is known as a
"deprecation cycle", and makes changes significantly easier to handle both for users
of multimodemodel, and for developers of other libraries that depend on multimodemodel.


.. _contributing.ci:

Testing With Continuous Integration
-----------------------------------

The *multimodemodel* test suite runs automatically the
`Gitlab CI/CD <https://git.geomar.de/help/ci/quick_start/index.md>`__,
continuous integration service, once your pull request is submitted.

A pull-request will be considered for merging when you have an all 'green' build. If any
tests are failing, then you will get a red 'X', where you can click through to see the
individual failed tests.

.. note::

   Each time you push to your PR branch, a new run of the tests will be
   triggered on the CI. If they haven't already finished, tests for any older
   commits on the same branch will be automatically cancelled.


.. _contributing.tdd:

Test-driven development/code writing
------------------------------------

*multimodemodel* is serious about testing and strongly encourages contributors to embrace
`test-driven development (TDD) <http://en.wikipedia.org/wiki/Test-driven_development>`_.
This development process "relies on the repetition of a very short development cycle:
first the developer writes an (initially failing) automated test case that defines a desired
improvement or new function, then produces the minimum amount of code to pass that test."
So, before actually writing any code, you should write your tests.  Often the test can be
taken from the original GitHub issue.  However, it is always worth considering additional
use cases and writing corresponding tests.

Adding tests is one of the most common requests after code is pushed to *multimodemodel*.  Therefore,
it is worth getting in the habit of writing tests ahead of time so that this is never an issue.

Like many packages, *multimodemodel* uses `pytest
<http://doc.pytest.org/en/latest/>`_.

Writing tests
~~~~~~~~~~~~~

All tests should go into the ``tests`` subdirectory of the specific package.
This folder contains many current examples of tests, and we suggest looking to these for
inspiration.

.. If your test requires working with files or
.. network connectivity, there is more information on the `testing page
.. <https://github.com/pydata/xarray/wiki/Testing>`_ of the wiki.

The easiest way to verify that your code is correct is to
explicitly construct the result you expect, then compare the actual result to
the expected correct result

.. code-block:: python

    def test_constructor_from_0d():
        expected = Dataset({None: ([], 0)})[None]
        actual = DataArray(0)
        assert_identical(expected, actual)


Using ``pytest``
~~~~~~~~~~~~~~~~

Here is an example of a self-contained set of tests that illustrate multiple
features that we like to use.

- functional style: tests are like ``test_*`` and *only* take arguments that are either
  fixtures or parameters
- ``pytest.mark`` can be used to set metadata on test functions, e.g. ``skip`` or ``xfail``.
- using ``parametrize``: allow testing of multiple cases
- to set a mark on a parameter, ``pytest.param(..., marks=...)`` syntax should be used
- ``fixture``, code for object construction, on a per-test basis
- using bare ``assert`` for scalars and truth-testing
- the typical pattern of constructing an ``expected`` and comparing versus the ``result``

We would name this file ``test_cool_feature.py`` and put in an appropriate place in the
``multimodemodel/tests/`` structure.

.. TODO: confirm that this actually works

.. code-block:: python

    import pytest
    import numpy as np
    import multimodemodel as mmm


    @pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64"])
    def test_dtypes(dtype):
        assert str(np.dtype(dtype)) == dtype

    @pytest.mark.parametrize(
        "dtype",
        [
            "float32",
            pytest.param("int16", marks=pytest.mark.skip),
            pytest.param(
                "int32", marks=pytest.mark.xfail(reason="to show how it works")
            ),
        ],
    )
    def test_mark(dtype):
        assert str(np.dtype(dtype)) == "float32"


    @pytest.fixture
    def dataarray():
        return xr.DataArray([1, 2, 3])


    @pytest.fixture(params=["int8", "int16", "int32", "int64"])
    def dtype(request):
        return request.param


    def test_series(dataarray, dtype):
        result = dataarray.astype(dtype)
        assert result.dtype == dtype

        expected = xr.DataArray(np.array([1, 2, 3], dtype=dtype))
        assert_equal(result, expected)



A test run of this yields

.. code-block:: shell

   ((multimodemodel) $ pytest test_cool_feature.py -v
    =============================== test session starts ================================
    platform darwin -- Python 3.6.4, pytest-3.2.1, py-1.4.34, pluggy-0.4.0 --
    cachedir: ../../.cache
    plugins: cov-2.5.1, hypothesis-3.23.0
    collected 11 items

    test_cool_feature.py::test_dtypes[int8] PASSED
    test_cool_feature.py::test_dtypes[int16] PASSED
    test_cool_feature.py::test_dtypes[int32] PASSED
    test_cool_feature.py::test_dtypes[int64] PASSED
    test_cool_feature.py::test_mark[float32] PASSED
    test_cool_feature.py::test_mark[int16] SKIPPED
    test_cool_feature.py::test_mark[int32] xfail
    test_cool_feature.py::test_series[int8] PASSED
    test_cool_feature.py::test_series[int16] PASSED
    test_cool_feature.py::test_series[int32] PASSED
    test_cool_feature.py::test_series[int64] PASSED

    ================== 9 passed, 1 skipped, 1 xfailed in 1.83 seconds ==================

Tests that we have ``parametrized`` are now accessible via the test name, for
example we could run these with ``-k int8`` to sub-select *only* those tests
which match ``int8``.


.. code-block:: shell

   ((multimodemodel) bash-3.2$ pytest  test_cool_feature.py  -v -k int8
   =========================== test session starts ===========================
   platform darwin -- Python 3.6.2, pytest-3.2.1, py-1.4.31, pluggy-0.4.0
   collected 11 items

   test_cool_feature.py::test_dtypes[int8] PASSED
   test_cool_feature.py::test_series[int8] PASSED


Running the test suite
----------------------

The tests can then be run directly inside your Git clone by typing::

    pytest multimodemodel

The tests suite is exhaustive and takes several seconds.  Often it is
worth running only a subset of tests first around your changes before running the
entire suite.

The easiest way to do this is with::

    pytest multimodemodel/path/to/test.py -k regex_matching_test_name

Or with one of the following constructs::

    pytest multimodemodel/tests/[test-module].py
    pytest multimodemodel/tests/[test-module].py::[TestClass]
    pytest multimodemodel/tests/[test-module].py::[TestClass]::[test_method]

For more, see the `pytest <http://doc.pytest.org/en/latest/>`_ documentation.

Running the performance test suite
----------------------------------

Performance matters and it is worth considering whether your code has introduced
performance regressions.  *multimodemodel* is starting to write a suite of benchmarking tests
using `asv <https://github.com/spacetelescope/asv>`__
to enable easy monitoring of the performance of critical *multimodemodel* operations.
These benchmarks are all found in the ``multimodemodel/asv_bench`` directory.  asv
supports both python2 and python3.

To use all features of asv, you will need either ``conda`` or
``virtualenv``. For more details please check the `asv installation
webpage <https://asv.readthedocs.io/en/latest/installing.html>`_.

To install asv::

    pip install git+https://github.com/spacetelescope/asv

If you need to run a benchmark, change your directory to ``asv_bench/`` and run::

    asv continuous -f 1.1 upstream/main HEAD

You can replace ``HEAD`` with the name of the branch you are working on,
and report benchmarks that changed by more than 10%.
The command uses ``conda`` by default for creating the benchmark
environments. If you want to use virtualenv instead, write::

    asv continuous -f 1.1 -E virtualenv upstream/main HEAD

The ``-E virtualenv`` option should be added to all ``asv`` commands
that run benchmarks. The default value is defined in ``asv.conf.json``.

Running the full benchmark suite can take up to one hour and use up a few GBs of RAM.
Usually it is sufficient to paste only a subset of the results into the pull
request to show that the committed changes do not cause unexpected performance
regressions.  You can run specific benchmarks using the ``-b`` flag, which
takes a regular expression.  For example, this will only run tests from a
``multimodemodel/asv_bench/benchmarks/groupby.py`` file::

    asv continuous -f 1.1 upstream/main HEAD -b ^groupby

If you want to only run a specific group of tests from a file, you can do it
using ``.`` as a separator. For example::

    asv continuous -f 1.1 upstream/main HEAD -b groupby.GroupByMethods

will only run the ``GroupByMethods`` benchmark defined in ``groupby.py``.

You can also run the benchmark suite using the version of *multimodemodel*
already installed in your current Python environment. This can be
useful if you do not have ``virtualenv`` or ``conda``, or are using the
``setup.py develop`` approach discussed above; for the in-place build
you need to set ``PYTHONPATH``, e.g.
``PYTHONPATH="$PWD/.." asv [remaining arguments]``.
You can run benchmarks using an existing Python
environment by::

    asv run -e -E existing

or, to use a specific Python interpreter,::

    asv run -e -E existing:python3.6

This will display stderr from the benchmarks, and use your local
``python`` that comes from your ``$PATH``.

Information on how to write a benchmark and how to use asv can be found in the
`asv documentation <https://asv.readthedocs.io/en/latest/writing_benchmarks.html>`_.

..
   TODO: uncomment once we have a working setup
         see https://github.com/pydata/multimodemodel/pull/5066

   The *multimodemodel* benchmarking suite is run remotely and the results are
   available `here <http://pandas.pydata.org/speed/multimodemodel/>`_.

Documenting your code
---------------------

Changes should be reflected in the release notes located in ``doc/whats-new.rst``.
This file contains an ongoing change log for each release.  Add an entry to this file to
document your fix, enhancement or (unavoidable) breaking change.  Make sure to include the
GitHub issue number when adding your entry (using ``:issue:`1234```, where ``1234`` is the
issue/pull request number).

If your code is an enhancement, it is most likely necessary to add usage
examples to the existing documentation.  This can be done following the section
regarding documentation :ref:`above <contributing.documentation>`.

Contributing your changes to *multimodemodel*
=============================================

Committing your code
--------------------

Keep style fixes to a separate commit to make your pull request more readable.

Once you've made changes, you can see them by typing::

    git status

If you have created a new file, it is not being tracked by git. Add it by typing::

    git add path/to/file-to-be-added.py

Doing 'git status' again should give something like::

    # On branch shiny-new-feature
    #
    #       modified:   /relative/path/to/file-you-added.py
    #

The following defines how a commit message should be structured:

    * A subject line with `< 72` chars.
    * One blank line.
    * Optionally, a commit message body.

Please reference the relevant issues in your commit message using ``#1234``.

Now you can commit your changes in your local repository::

    git commit -m

Pushing your changes
--------------------

When you want your changes to appear publicly on your Gitlab page, push your
forked feature branch's commits::

    git push origin shiny-new-feature

Here ``origin`` is the default name given to your remote repository on Gitlab.
You can see the remote repositories::

    git remote -v

Now your code is on Gitlab, but it is not yet a part of the *multimodemodel* project.
For that to happen, a pull request needs to be submitted on Gitlab.

Review your code
----------------

When you're ready to ask for a code review, file a pull request. Before you do, once
again make sure that you have followed all the guidelines outlined in this document
regarding code style, tests, performance tests, and documentation. You should also
double check your branch changes against the branch it was based on:

#. Navigate to your repository on Gitlab
#. Click on ``Branches``
#. Click on the ``Compare`` button for your feature branch
#. Select the ``base`` and ``compare`` branches, if necessary. This will be ``master`` and
   ``shiny-new-feature``, respectively.

Finally, make the pull request
------------------------------

If everything looks good, you are ready to make a pull request.  A pull request is how
code from a local repository becomes available to the Gitlab community and can be looked
at and eventually merged into the ``master`` version.  This pull request and its associated
changes will eventually be committed to the ``master`` branch and available in the next
release.  To submit a pull request:

#. Navigate to your repository on Gitlab
#. Click on the ``Pull Request`` button
#. You can then click on ``Commits`` and ``Files Changed`` to make sure everything looks
   okay one last time
#. Write a description of your changes in the ``Preview Discussion`` tab
#. Click ``Send Pull Request``.

This request then goes to the repository maintainers, and they will review
the code. If you need to make more changes, you can make them in
your branch, add them to a new commit, push them to GitHub, and the pull request
will automatically be updated.  Pushing them to GitHub again is done by::

    git push origin shiny-new-feature

This will automatically update your pull request with the latest code and restart the
:ref:`Continuous Integration <contributing.ci>` tests.


PR checklist
------------

- **Properly comment and document your code.** See :ref:`"Documenting your code" <Documenting your code>`.
- **Test that the documentation builds correctly** by typing ``make html`` in the ``doc`` directory. This is not strictly necessary, but this may be easier than waiting for CI to catch a mistake. See :ref:`"Contributing to the documentation" <Contributing to the documentation>`.
- **Test your code**.

    - Write new tests if needed. See :ref:`Test-driven development/code writing`.
    - Test the code using `Pytest <http://doc.pytest.org/en/latest/>`_. Running all tests (type ``pytest`` in the root directory) takes a while, so feel free to only run the tests you think are needed based on your PR (example: ``pytest multimodemodel/tests/test_grid.py``). CI will catch any failing tests.

- **Properly format your code** and verify that it passes the formatting guidelines set by `Black <https://black.readthedocs.io/en/stable/>`_ and `Flake8 <http://flake8.pycqa.org/en/latest/>`_. See :ref:`"Code formatting" <Code formatting>` to run these automatically on each commit.

    - Run ``pre-commit run --all-files`` in the root directory. This may modify some files. Confirm and commit any formatting changes.

- **Push your code and** `create a PR on Gitlab <https://git.geomar.de/help/user/project/merge_requests/index.md>`_.
- **Use a helpful title for your pull request** by summarizing the main contributions rather than using the latest commit message. If the PR addresses an `issue <https://git.geomar.de/mcgroup/multimode-model/-/issues>`_, please `reference it <https://git.geomar.de/help/user/markdown.md#gitlab-specific-references>`_.
