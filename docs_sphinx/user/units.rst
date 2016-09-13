Physical units
==============

.. contents::
    :local:
    :depth: 1

Brian includes a system for defining physical units. These are defined by
their standard SI unit names: amp,
kilogram, second, metre/meter, mole and the derived units coulomb, farad,
gram/gramme, hertz, joule, pascal, ohm,  siemens, volt, watt, together with
prefixed versions (e.g. ``msiemens = 0.001*siemens``) using the prefixes
p, n, u, m, k, M, G, T (two exceptions: kilogram is not imported with any
prefixes, metre and meter are additionaly defined with the "centi" prefix,
i.e. cmetre/cmeter). In addition a couple of useful standard abbreviations like
"cm" (instead of cmetre/cmeter), "nS" (instead of nsiemens),
"ms" (instead of msecond), "Hz" (instead of hertz), etc. are included.


Using units
-----------
You can generate a physical quantity by multiplying a scalar or vector value
with its physical unit::

    >>> tau = 20*ms
    >>> print tau
    20. ms
    >>> rates = [10, 20, 30] * Hz
    >>> print rates
    [ 10.  20.  30.] Hz

Brian will check the consistency of operations on units and raise an error for
dimensionality mismatches::

    >>> tau += 1  # ms? second?  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    DimensionMismatchError: Cannot calculate ... += 1, units do not match (units are second and 1).
    >>> 3*kgram + 3*amp   # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    DimensionMismatchError: Cannot calculate 3. kg + 3. A, units do not match (units are kgramme and amp).

Most Brian functions will also complain about non-specified or incorrect units::

    >>> G = NeuronGroup(10, 'dv/dt = -v/tau: volt', dt=0.5)   # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    DimensionMismatchError: Function "__init__" expected a quantitity with unit second for argument "dt" but got 0.5 (unit is 1).

Numpy functions have been overwritten to correctly work with units (see the
:doc:`developer documentation <../developer/units>` for more details)::

    >>> print mean(rates)
    20. Hz
    >>> print rates.repeat(2)
    [ 10.  10.  20.  20.  30.  30.] Hz

Removing units
--------------
There are various options to remove the units from a value (e.g. to use it with
analysis functions that do not correctly work with units)

* Divide the value by its unit (most of the time the recommended option
  because it is clear about the scale)
* Transform it to a pure numpy array in the base unit by calling `asarray`
  (no copy) or `array` (copy)
* Directly get the unitless value of a state variable by appending an underscore
  to the name

::

    >>> tau/ms
    20.0
    >> asarray(rates)
    array([ 10.,  20.,  30.])
    >>> G = NeuronGroup(5, 'dv/dt = -v/tau: volt')
    >>> print G.v_[:]
    [ 0.,  0.,  0.,  0.,  0.]


.. admonition:: The following topics are not essential for beginners.

    |

Importing units
---------------
Brian generates standard names for units, combining the unit name (e.g.
"siemens") with a prefixes (e.g. "m"), and also generates squared and cubed
versions by appending a number. For example, the units "msiemens", "siemens2",
"usiemens3" are all predefined. You can import these units from the package
``brian2.units.allunits`` -- accordingly, an
``from brian2.units.allunits import *`` will result in everything from
``Ylumen3`` (cubed yotta lumen) to ``ymol`` (yocto mole) being imported.

A better choice is normally to do ``from brian2.units import *`` or import
everything ``from brian2 import *``, this imports only the base units amp,
kilogram, second, metre/meter, mole and the derived units coulomb, farad,
gram/gramme, hertz, joule, pascal, ohm,  siemens, volt, watt, together with the
prefixes p, n, u, m, k, M, G, T (two exceptions: kilogram is not imported with
any prefixes, metre and meter are additionaly defined with the "centi" prefix,
i.e. cmetre/cmeter).

In addition a couple of useful standard abbreviations like
"cm" (instead of cmetre/cmeter), "nS" (instead of nsiemens),
"ms" (instead of msecond), "Hz" (instead of hertz), etc. are added (they can
be individually imported from ``brian2.units.stdunits``).

In-place operations on quantities
---------------------------------
In-place operations on quantity arrays change the underlying array, in the
same way as for standard numpy arrays. This means, that any other variables
referencing the same object will be affected as well::

    >>> q = [1, 2] * mV
    >>> r = q
    >>> q += 1*mV
    >>> q
    array([ 2.,  3.]) * mvolt
    >>> r
    array([ 2.,  3.]) * mvolt

In contrast, scalar quantities will never change the underlying value but
instead return a new value (in the same way as standard Python scalars)::

    >>> x = 1*mV
    >>> y = x
    >>> x *= 2
    >>> x
    2. * mvolt
    >>> y
    1. * mvolt

