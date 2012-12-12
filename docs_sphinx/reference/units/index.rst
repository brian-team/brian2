.. currentmodule:: brian2

Unit system
===========

The main classes for working with units (although they are rarely used
explicitly by users):

.. autosummary:: Quantity
   :toctree:
.. autosummary:: Unit
   :toctree:   
   
.. autoexception:: DimensionMismatchError

A decorator that can be used for unit checking:

.. autofunction:: brian2.units.fundamentalunits.check_units

Advanced features
-----------------

Classes used internally
~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary:: brian2.units.fundamentalunits.Dimension
   :toctree:
.. autosummary:: brian2.units.fundamentalunits.UnitRegistry
   :toctree:

Utility functions
~~~~~~~~~~~~~~~~~
*Display/create quantities*

.. autofunction:: brian2.units.fundamentalunits.display_in_unit
.. autofunction:: brian2.units.fundamentalunits.quantity_with_dimensions

*Function wrappers*

.. autofunction:: brian2.units.fundamentalunits.wrap_function_dimensionless
.. autofunction:: brian2.units.fundamentalunits.wrap_function_keep_dimensions
.. autofunction:: brian2.units.fundamentalunits.wrap_function_change_dimensions
.. autofunction:: brian2.units.fundamentalunits.wrap_function_remove_dimensions

*Create/compare dimensions*

.. autoattribute:: brian2.units.fundamentalunits.DIMENSIONLESS
.. autofunction:: brian2.units.fundamentalunits.get_or_create_dimension
.. autofunction:: brian2.units.fundamentalunits.is_scalar_type
.. autofunction:: brian2.units.fundamentalunits.get_dimensions
.. autofunction:: brian2.units.fundamentalunits.is_dimensionless
.. autofunction:: have_same_dimensions
.. autofunction:: brian2.units.fundamentalunits.fail_for_dimension_mismatch

*Unit registry*

.. autofunction:: register_new_unit
.. autofunction:: brian2.units.fundamentalunits.all_registered_units
.. autofunction:: brian2.units.fundamentalunits.get_unit
.. autofunction:: brian2.units.fundamentalunits.get_unit_fast
