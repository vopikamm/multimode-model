.. currentmodule:: multimodemodel

What's New
==========

.. _whats-new.0.1.0:

v0.1.0 (unreleased)
---------------------

*Initial realease*

New Features
~~~~~~~~~~~~
- Support for splitting the domain into subdomains and distribute the computation across several processes.
- Use `flake8 <https://flake8.pycqa.org/en/latest/>`_ for linting and `black <https://black.readthedocs.io/en/stable/>`_ for styling.
- Enable `SAST <https://git.geomar.de/help/user/application_security/sast/index.md>`_.
- Support of grid staggering via :py:class:`.StaggeredGrid` class.
- Formulate solver to work on curvilinear coordinates
- Support of several time integration methods, such as :py:func:`.euler_forward`, :py:func:`.adams_bashforth2` and :py:func:`.adams_bashforth3`
- Support of `None` type data attributes of :py:class:`.Variable`.
- Provide a benchmark suite to track performance Changes
- `xarray.DataArray` view of a :py:class:`.Variable`.

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~

Documentation
~~~~~~~~~~~~~
- Online documentation rendered with `sphinx <https://www.sphinx-doc.org/en/master/>`_.

Internal Changes
~~~~~~~~~~~~~~~~
