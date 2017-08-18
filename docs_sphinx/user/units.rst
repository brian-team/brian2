Physical units
==============

.. contents::
    :local:
    :depth: 1

Brian includes a system for physical units. The base units are defined by their
standard SI unit names: ``amp``/``ampere``, ``kilogram``/``kilogramme``,
``second``, ``metre``/``meter``, ``mole``/``mol``, ``kelvin``, and ``candela``.
In addition to these base units, Brian defines a set of derived units:
``coulomb``, ``farad``, ``gram``/``gramme``, ``hertz``, ``joule``, ``liter``/
``litre``, ``molar``, ``pascal``, ``ohm``,  ``siemens``, ``volt``, ``watt``,
together with prefixed versions (e.g. ``msiemens = 0.001*siemens``) using the
prefixes ``p, n, u, m, k, M, G, T`` (two exceptions to this rule: ``kilogram``
is not defined with any additional prefixes, and ``metre`` and ``meter`` are
additionaly defined with the "centi" prefix, i.e. ``cmetre``/``cmeter``).
For convenience, a couple of additional useful standard abbreviations such as
``cm`` (instead of ``cmetre``/``cmeter``), ``nS`` (instead of ``nsiemens``),
``ms`` (instead of ``msecond``), ``Hz`` (instead of ``hertz``), ``mM``
(instead of ``mmolar``) are included. To avoid clashes with common variable
names, no one-letter abbreviations are provided (e.g. you can use ``mV`` or
``nS``, but *not* ``V`` or ``S``).

Using units
-----------
You can generate a physical quantity by multiplying a scalar or vector value
with its physical unit::

    >>> tau = 20*ms
    >>> print(tau)
    20. ms
    >>> rates = [10, 20, 30]*Hz
    >>> print(rates)
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
    DimensionMismatchError: Cannot calculate 3. kg + 3. A, units do not match (units are kilogram and amp).

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


Temperatures
------------
Brian only supports temperatures defined in °K, using the provided ``kelvin``
unit object. Other conventions such as °C, or °F are not compatible with Brian's
unit system, because they cannot be expressed as a multiplicative scaling of the
SI base unit kelvin (their zero point is different). However, in biological
experiments and modeling, temperatures are typically reported in °C. How to use
such temperatures depends on whether they are used as *temperature differences*
or as *absolute temperatures*:

temperature differences
    Their major use case is the correction of time constants for differences in
    temperatures based on the `Q10 temperature coefficient <https://en.wikipedia.org/wiki/Q10_(temperature_coefficient)>`_.
    In this case, all temperatures can directly use ``kelvin`` even though the
    temperatures are reported in Celsius, since temperature differences in
    Celsius and Kelvin are identical.

absolute temperatures
    Equations such as the `Goldman–Hodgkin–Katz voltage equation <https://en.wikipedia.org/wiki/Goldman_equation>`_
    have a factor that depends on the absolute temperature measured in Kelvin.
    To get this temperature from a temperature reported in °C, you can use the
    ``zero_celsius`` constant from the `brian2.units.constants` package (see
    below)::

        from brian2.units.constants import zero_celsius

        celsius_temp = 27
        abs_temp = celsius_temp*kelvin + zero_celsius

.. note:: Earlier versions of Brian had a ``celsius`` unit which was in fact
          identical to ``kelvin``. While this gave the correct results for
          temperature differences, it did not correctly work for absolute
          temperatures. To avoid confusion and possible misinterpretation,
          the ``celsius`` unit has therefore been removed.

.. _constants:

Constants
---------
The `brian2.units.constants` package provides a range of physical constants that
can be useful for detailed biological models. Brian provides the following
constants:

==================== ================== ======================= ==================================================================
Constant             Symbol(s)          Brian name              Value
==================== ================== ======================= ==================================================================
Avogadro constant    :math:`N_A, L`     ``avogadro_constant``   :math:`6.022140857\times 10^{23}\,\mathrm{mol}^{-1}`
Boltzmann constant   :math:`k`          ``boltzmann_constant``  :math:`1.38064852\times 10^{-23}\,\mathrm{J}\,\mathrm{K}^{-1}`
Electric constant    :math:`\epsilon_0` ``electric_constant``   :math:`8.854187817\times 10^{-12}\,\mathrm{F}\,\mathrm{m}^{-1}`
Electron mass        :math:`m_e`        ``electron_mass``       :math:`9.10938356\times 10^{-31}\,\mathrm{kg}`
Elementary charge    :math:`e`          ``elementary_charge``   :math:`1.6021766208\times 10^{-19}\,\mathrm{C}`
Faraday constant     :math:`F`          ``faraday_constant``    :math:`96485.33289\,\mathrm{C}\,\mathrm{mol}^{-1}`
Gas constant         :math:`R`          ``gas_constant``        :math:`8.3144598\,\mathrm{J}\,\mathrm{mol}^{-1}\,\mathrm{K}^{-1}`
Magnetic constant    :math:`\mu_0`      ``magnetic_constant``   :math:`12.566370614\times 10^{-7}\,\mathrm{N}\,\mathrm{A}^{-2}`
Molar mass constant  :math:`M_u`        ``molar_mass_constant`` :math:`1\times 10^{-3}\,\mathrm{kg}\,\mathrm{mol}^{-1}`
0°C                                     ``zero_celsius``        :math:`273.15\,\mathrm{K}`
==================== ================== ======================= ==================================================================

Note that these constants are not imported by default, you will have to
explicitly import them from `brian2.units.constants`. During the import, you
can also give them shorter names using Python's ``from ... import ... as ...``
syntax. For example, to calculate the :math:`\frac{RT}{F}` factor that appears
in the `Goldman–Hodgkin–Katz voltage equation <https://en.wikipedia.org/wiki/Goldman_equation>`_
you can use::

    from brian2 import *
    from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F

    celsius_temp = 27
    T = celsius_temp*kelvin + zero_celsius
    factor = R*T/F


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
everything ``from brian2 import *`` which only imports the units mentioned in
the introductory paragraph (base units, derived units, and some standard
abbreviations).

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
