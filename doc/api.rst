.. currentmodule:: multimodemodel

#############
API Reference
#############

This page provides an auto-generated summary of multimodemodel's API

.. autosummary::
   :toctree: generated/

   Parameter
   Parameter.f
   Parameter.split
   Parameter.merge
   Parameter.__eq__
   Variable
   Variable.data
   Variable.grid
   Variable.time
   Variable.split
   Variable.merge
   Variable.copy
   Variable.as_dataarray
   Variable.safe_data
   Variable.__eq__
   Variable.__add__
   State
   State.split
   State.merge
   State.__add__
   StateDeque
   StateDeque.split
   StateDeque.merge
   Domain
   Domain.increment_iteration
   Domain.split
   Domain.merge
   Domain.copy
   Domain.__eq__
   Grid
   Grid.ndim
   Grid.shape
   Grid.dim_x
   Grid.dim_y
   Grid.dim_z
   Grid.cartesian
   Grid.regular_lat_lon
   Grid.__eq__
   StaggeredGrid
   StaggeredGrid.split
   StaggeredGrid.merge
   StaggeredGrid.cartesian_c_grid
   StaggeredGrid.regular_lat_lon_c_grid
   GridShift
   RegularSplitMerger
   BorderSplitter
   BorderMerger
   Border
   Tail
   integrate
   linearised_SWE
   TimeSteppingFunction
   StateIncrement
   time_stepping_function
   euler_forward
   adams_bashforth2
   adams_bashforth3
   Solver
   Solver.rhs
   Solver.integrate
   Solver.integrate_border
   Solver.get_border_width
   coriolis_i
   coriolis_j
   divergence_j
   divergence_i
   pressure_gradient_j
   pressure_gradient_i
   sum_states
   sum_vars
   linear_combination
   CoriolisFunc
   f_constant
   beta_plane
   f_on_sphere
