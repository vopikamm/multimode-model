.. MultiModeModel documentation master file, created by
   sphinx-quickstart on Mon Sep 13 15:09:02 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MultiModeModel: Non-linear ocean model using separation into vertical normal modes
==================================================================================

**multimodemodel** is an open source project and Python package
that allows to simulate the dynamics of a flat bottom ocean by
expressing the state of the ocean by an expansion in terms of vertical
normal modes. To learn more about the physical background, take a look
at the :ref:`physical basis` section.

Multimodemodel expresses the state of the system as a :py:class:`.State` object.
The various physical processes are represented by functions that take the current
:py:class:`.State` and :py:class:`.Parameters` and return the respective tendency.
These functions are Just-in-time-compiled by Numba_.

A list of examples can be found in the :doc:`examples` section.

.. _Numba: https://numba.pydata.org/


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: For users

   Getting Started <getting_started/index>
   Model formulation <model_formulation>
   Examples <examples>
   API reference <api>


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: For developers/contributors

   Contributing guide <contributing>
   What's new <whats_new>
   Gitlab repository <https://git.geomar.de/mcgroup/multimodemodel>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
