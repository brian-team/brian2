Coding conventions
==================
General recommendations
-----------------------
Syntax is chosen as much as possible from the user point of view,
to reflect the concepts as directly as possible. Ideally, a Brian script
should be readable by someone who doesn't know Python or Brian, although this
isn't always possible. Function, class and keyword argument names should be
explicit rather than abbreviated and consistent across Brian. See Romain's paper 
`On the design of script languages for neural simulators
<http://briansimulator.org/WordPress/wp-content/uploads/2012/05/On-the-design-of-script-languages-for-neural-simulation.pdf>`__ 
for a discussion.

We use the `PEP-8 coding conventions <https://www.python.org/dev/peps/pep-0008/>`__
for our code. This in particular includes the following conventions:

* Use 4 spaces instead of tabs per indentation level
* Use spaces after commas and around the following binary operators:
  assignment (=), augmented assignment (+=, -= etc.),
  comparisons (==, <, >, !=, <>, <=, >=, in, not in, is, is not), 
  Booleans (and, or, not).
* Do *not* use spaces around the equals sign in keyword arguments or when
  specifying default values. Neither put spaces immediately inside parentheses,
  brackets or braces, immediately before the open parenthesis that starts the
  argument list of a function call, or immediately before the open parenthesis
  that starts an indexing or slicing.
* Avoid using a backslash for continuing lines whenever possible, instead use
  Python's implicit line joining inside parentheses, brackets and braces.
* The core code should only contain ASCII characters, no encoding has to be declared
* imports should be on different lines (e.g. do not use ``import sys, os``) and
  should be grouped in the following order, using blank lines between each group:
  
  	1. standard library imports
  	2. third-party library imports (e.g. numpy, scipy, sympy, ...)
  	3. brian imports

* Use absolute imports for everything outside of "your" package, e.g. if you
  are working in `brian2.equations`, import functions from the stringtools
  modules via ``from brian2.utils.stringtools import ...``. Use the full path
  when importing, e.g. do ``from brian2.units.fundamentalunits import seconds``
  instead of ``from brian2 import seconds``.
* Use "new-style" relative imports for everything in "your" package, e.g. in
  ``brian2.codegen.functions.py`` import the `Function` class as 
  ``from .specifiers import Function``.  
* Do not use wildcard imports (``from brian2 import *``), instead import only the
  identifiers you need, e.g. ``from brian2 import NeuronGroup, Synapses``. For 
  packages like numpy that are used a lot, use ``import numpy as np``. But
  note that the user should still be able to do something like
  ``from brian2 import *`` (and this style can also be freely used in examples
  and tests, for example). Modules always have to use the ``__all__`` mechanism
  to specify what is being made available with a wildcard import. As an
  exception from this rule, the main ``brian2/__init__.py`` may use wildcard
  imports.
