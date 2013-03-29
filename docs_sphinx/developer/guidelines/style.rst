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

We use the `PEP-8 coding conventions <http://www.python.org/dev/peps/pep-0008/>`__
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

Python 2 vs. Python 3
---------------------
Brian is written in Python 2 but runs on Python 3 using the
`2to3 <http://docs.python.org/2/library/2to3.html>`__ conversion tool (which is
automatically applied if Brian is installed using the standard
``python setup.py install`` mechanism). To make this possible without too much
effort, Brian no longer supports Python 2.5 and can therefore make use of a
couple of forward-compatible (but backward-incompatible) idioms introduced in
Python 2.6.  The `Porting to Python 3 <http://python3porting.com/>`__
book is available online and has a lot of information on these topics. Here are
some things to keep in mind when developing Brian:

* If you are working with integers and using division, consider using ``//``
  for flooring division (default behaviour for ``/`` in python 2) and switch the
  behaviour of ``/`` to floating point division by using
  ``from __future__ import division`` .
* If importing modules from the standard library (which have changed quite a
  bit from Python 2 to Python 3), only use simple import statements like
  ``import itertools`` instead of ``from itertools import izip`` -- *2to3* is
  otherwise unable to make the correct conversion.
* If you are using the ``print`` statement (which should only occur in tests,
  in particular doctests -- always use the :doc:`logging` framework if you want
  to present messages to the user otherwise), try "cheating" and use the
  functional style in Python 2, i.e. write ``print('some text')`` instead of
  ``print 'some text'``. More complicated print statements should be avoided,
  e.g instead of ``print >>sys.stderr, 'Error message`` use
  ``sys.stderr.write('Error message\n')`` (or, again, use logging).
* Exception stacktraces look a bit different in Python 2 and 3: For non-standard
  exceptions, Python 2 only prints the Exception class name (e.g.
  ``DimensionMismatchError``) whereas Python 3 prints the name including the
  module name (e.g. ``brian2.units.fundamentalunits.DimensionMismatchError``).
  This will make doctests fail that match the exception message. In this case,
  write the doctest in the style of Python 2 but add the doctest directive
  ``#doctest: +IGNORE_EXCEPTION_DETAIL`` to the statement leading to the
  exception. This unfortunately has the side effect of also ignoring the
  text of the exception, but it will still fail for an incorrect exception type.
* If you write code reading and writing strings to files, make sure you make
  the distinction between bytes and unicode (see `"separate binary data and strings" <http://python3porting.com/preparing.html#separate-binary-data-and-strings>`__ )
  In general, strings within Brian are unicode strings and only converted to
  bytes when reading from or writing to a file (or something like a network
  stream, for example).
* If you are sorting lists or dictionaries, have a look at
  `"when sorting, use key instead of cmp" <http://python3porting.com/preparing.html#when-sorting-use-key-instead-of-cmp>`__  
* Make sure to define a ``__hash__`` function for objects that define an
  ``__eq__`` function (and to define it consistently). Python 3 is more strict
  about this, an object with ``__eq__`` but without ``__hash__`` is unhashable.
