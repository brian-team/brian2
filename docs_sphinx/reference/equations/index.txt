.. currentmodule:: brian2

Equations and code strings
==========================

Most of the time, the user does not explicitly construct `Expression`,
`Statements` or `Equations` objects. Instead, directly pass a string to the
constructor of NeuronGroup, Synapses, etc. The following classes are only useful
if you want to have full control over the namespace in which the code strings are
interpreted.

.. autosummary:: Expression
   :toctree:
.. autosummary:: Statements
   :toctree:
.. autosummary:: Equations
   :toctree:

For internal use only
---------------------
.. autosummary:: brian2.equations.codestrings.CodeString
   :toctree:
.. autosummary:: brian2.equations.equations.SingleEquation
   :toctree:

Parsing
~~~~~~~
.. autofunction:: brian2.equations.unitcheck.get_unit_from_string
.. autofunction:: brian2.equations.equations.parse_string_equations

Namespaces
~~~~~~~~~~
.. autofunction:: brian2.equations.unitcheck.get_default_unit_namespace
.. autofunction:: brian2.equations.codestrings.get_default_numpy_namespace

Checking identifiers
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: brian2.equations.equations.check_identifier
.. autofunction:: brian2.equations.equations.check_identifier_basic
.. autofunction:: brian2.equations.equations.check_identifier_reserved
