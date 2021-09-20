Installation
============

Required dependencies
---------------------

- Python (3.7 or later)
- `numba <https://numba.pydata.org/>`__ (0.50.1 or later)
- `numpy <http://www.numpy.org/>`__

Optional dependencies
---------------------

.. note::

  When using pip to install multimodemodel, optional dependencies can be installed by
  specifying *extras*. :ref:`Instructions` are given below.


Instructions
------------

First clone the repository::

    $ git clone https://git.geomar.de/mcgroup/multimode-model.git

Switch to the directory and install multimodemodel with::

    $ cd multimodemodel
    $ pip install .

Note that for development you can also install multimodemodel with the `editable` flag::

    $ pip install -e .

We also maintain other dependency sets for additional subsets of functionality::

    $ pip install .[xarray]    # Install optional dependencies for conversion to xarray
    $ pip install .[test]      # Install optional dependencies for testing
    $ pip install .[lint]      # Install optional dependencies for linting
    $ pip install .[examples]  # Install optional dependencies for examples
    $ pip install .[complete]  # Install all the above
    $ pip install .[docs]      # Install all above plus dependencies for building the documentation

The above commands install the `optional dependencies`_. To know which dependencies
would be installed, take a look at the ``[options.extras_require]`` section in
``setup.cfg``:

.. literalinclude:: ../../setup.cfg
   :language: ini
   :start-at: [options.extras_require]
   :end-before: [tool:pytest]


Testing
-------

To run the test suite after installing multimodemodel, install (via pypi or conda) `py.test
<https://pytest.org>`__ and run ``pytest`` in the root directory of the package.
