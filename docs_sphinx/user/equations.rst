Equations
=========

.. contents::
    :local:
    :depth: 1

.. _equation_strings:

Equation strings
----------------
Equations are used both in `NeuronGroup` and `Synapses` to:

* define state variables
* define continuous-updates on these variables, through differential equations

.. note::

    Brian models are defined by systems of first order ordinary differential
    equations, but you might see the integrated form of synapses in some textbooks
    and papers. See :doc:`converting_from_integrated_form` for details on how
    to convert between these representations.

Equations are defined by multiline strings, where each line takes of one of three forms:

1. ``dx/dt = f : unit (flags)`` (differential equation)
2. ``x = f : unit (flags)`` (subexpression)
3. ``x : unit (flags)`` (parameter)

Each of these definitions can take optional ``flags`` in parentheses after the ``unit`` declaration (see :ref:`flags` below).

The first form defines a differential equation that determines how a variable evolves over time.
The second form defines a subexpression, which is useful to make complex equations more readable, and to have a
name for expressions that can be recorded with a `StateMonitor`. Such subexpressions are computed
"on demand" and are not stored. Their use is therefore mostly for convenience and does not affect simulation time
or memory usage. The third form defines a parameter, which is a value that is unique to each neuron or synapse.
Its value can either be constant (e.g. to have a heterogeneous population of neurons) or can be a value that gets
updated by synaptic events, or by `~.Group.run_regularly` operations.

Each definition may be spread out over multiple lines to improve readability, and can include comments after ``#``.
The ``unit`` definition defines the dimension of the variable. Note that these are always the dimensions of the
*variable* defined in the line, even in the case of differential equations. Therefore, the unit for the membrane
potential would be ``volt`` and not ``volt/second`` (the dimensions of its derivative). The ``unit`` always has to
be a *base unit*, i.e., one must write ``volt``, not ``mV``. This is to make it clear that the values are
internally always saved in the base units, so no confusion can arise when getting the values out of a `NeuronGroup`
and discarding the units. Compound units are of course allowed as well (e.g. ``farad/meter**2``).
There are also three special "units" that can be used: ``1`` denotes a dimensionless floating point variable,
``boolean`` and ``integer`` denote dimensionless variables of the respective kind.

.. note:: For molar concentration, the base unit that has to be used in the equations is ``mmolar`` (or ``mM``), *not*
          ``molar``. This is because 1 molar is 10³ mol/m³ in SI units (i.e., it has a "scale" of 10³), whereas
          1 millimolar corresponds to 1 mol/m³.

Arithmetic operations and functions
-----------------------------------
Equation strings can make use of standard arithmetic operations for numerical
values, using the Python 3 syntax. The supported operations are ``+``, ``-``,
``*``, ``/`` (floating point division), ``//`` (flooring division), ``%``
(remainder), ``**`` (power). For variable assignments, e.g. in reset statements,
the corresponding in-place assignments such as ``+=`` can be used as well.
For comparisons, the operations ``==`` (equality), ``!=`` (inequality), ``<``,
``<=``, ``>``, and ``>=`` are available. Truth values can be combined using
``and`` and ``or``, or negated using ``not``. Note that Brian does not support
any operations specific to integers, e.g. "bitwise AND" or shift operations. Importantly, while equations use Python
syntax, they are not Python code; they are parsed and translated to the target language by Brian, and can therefore
not use arbitrary Python syntax or functions. They are also written in a "for each neuron/synapse" style, so their
interpretation depends on the context in which they are used. For example, when a synaptic pre/post statement refers to a
variable of a pre- or post-synaptic neurons, it only refers to the subset of neurons that spiked. This also means that
you cannot (and usually don't need to) use Python's indexing syntax to refer to specific elements of a group.

.. warning::

    Brian versions up to 2.1.3.1 did not support ``//`` as the floor division
    operator and potentially used different semantics for the ``/`` operator
    depending on whether Python 2 or 3 was used. To write code that correctly
    and unambiguously works with both newer and older Brian versions, you can
    use expressions such as ``1.0*a/b`` to enforce floating point division (if
    one of the operands is a floating point number, both Python 2 and 3 will use
    floating point division), or ``floor(a/b)`` to enforce flooring division
    Note that the ``floor`` function always returns a floating point value, if
    it is important that the result is an integer value, additionally wrap it
    with the ``int`` function.

Brian also supports standard mathematical functions with the same names as used
in the ``numpy`` library (e.g. ``exp``, ``sqrt``, ``abs``, ``clip``, ``sin``,
``cos``, ...) -- for a full list see :ref:`default_functions`. Note that support
for such functions is provided by Brian itself and the translation to the
various code generation targets is automatically taken care of. You should
therefore refer to them directly by name and not as e.g. ``np.sqrt`` or
``numpy.sqrt``, regardless of the way you
:doc:`imported Brian or numpy <import>`. This also means that you cannot
directly refer to arbitrary functions from ``numpy`` or other libraries. For
details on how to extend the support to non-default functions see
:ref:`user_functions`.

Special variables
-----------------

Some special variables are defined, e.g. `t`, `dt` (time) and `xi` (white noise). For a full list see :ref:`special_symbols` below.
Variable names starting with an underscore and a couple of other names that have special meanings under certain
circumstances (e.g. names ending in ``_pre`` or ``_post``) are forbidden.

For stochastic equations with several ``xi`` values it is necessary to make clear whether they correspond to the same
or different noise instantiations. To make this distinction, an arbitrary suffix can be used, e.g. using ``xi_1`` several times
refers to the same variable, ``xi_2`` (or ``xi_inh``, ``xi_alpha``, etc.) refers to another. An error will be raised if
you use more than one plain ``xi`` without any suffix. Note that noise is always independent across neurons, you can only work around this
restriction by defining your noise variable as a shared parameter and update it using a user-defined function (e.g. with `~Group.run_regularly`),
or create a group that models the noise and link to its variable (see :ref:`linked_variables`).

.. _external-variables:

External references
-------------------
Equations defining neuronal or synaptic equations can contain references to
external constants or functions. These references are looked up at the time
that the simulation is run. If you don't specify where to look them up, it
will look in the Python local/global namespace (i.e. the block of code where
you call `run`). If you want to override this, you can specify an explicit
"namespace". This is a Python dictionary with keys being variable names as
they appear in the equations, and values being the desired value of that
variable. This namespace can be specified either in the creation of the group
or when you can the `run` function using the ``namespace`` keyword argument.

The following three examples show the different ways of providing external
variable values, all having the same effect in this case::

	# Explicit argument to the NeuronGroup
	G = NeuronGroup(1, 'dv/dt = -v / tau : 1', namespace={'tau': 10*ms})
	net = Network(G)
	net.run(10*ms)

	# Explicit argument to the run function
	G = NeuronGroup(1, 'dv/dt = -v / tau : 1')
	net = Network(G)
	net.run(10*ms, namespace={'tau': 10*ms})

	# Implicit namespace from the context
	G = NeuronGroup(1, 'dv/dt = -v / tau : 1')
	net = Network(G)
	tau = 10*ms
	net.run(10*ms)

See :doc:`../advanced/namespaces` for more details.

.. admonition:: The following topics are not essential for beginners.

    |

.. _flags:

Flags
-----
A *flag* is a keyword in parentheses at the end of the line, which
qualifies the equations. There are several keywords:

*event-driven*
  this is only used in Synapses, and means that the differential equation should be updated
  only at the times of events. This implies that the equation is taken out of the continuous
  state update, and instead a event-based state update statement is generated and inserted into
  event codes (pre and post).
  This can only qualify differential equations of synapses. Currently, only one-dimensional
  linear equations can be handled (see below).
*unless refractory*
  this means the variable is not updated during the refractory period.
  This can only qualify differential equations of neuron groups.
*constant*
  this means the parameter will not be changed during a run. This allows
  optimizations in state updaters. This can only qualify parameters.
*constant over dt*
  this means that the subexpression will be only evaluated once at the beginning
  of the time step. This can be useful to e.g. approximate a non-linear term as
  constant over a time step in order to use the ``linear`` numerical integration
  algorithm. It is also mandatory for subexpressions that refer to stateful
  functions like ``rand()`` to make sure that they are only evaluated once
  (otherwise e.g. recording the value with a `StateMonitor` would re-evaluate it
  and therefore not record the same values that are used in other places). This
  can only qualify subexpressions.
*shared*
  this means that a parameter or subexpression is not neuron-/synapse-specific
  but rather a single value for the whole `NeuronGroup` or `Synapses`. A shared
  subexpression can only refer to other shared variables.
*linked*
  this means that a parameter refers to a parameter in another `NeuronGroup`.
  See :ref:`linked_variables` for more details.

Multiple flags may be specified as follows::

	dx/dt = f : unit (flag1,flag2)

.. _special_symbols:

List of special symbols
-----------------------

The following lists all of the special symbols that Brian uses in
equations and code blocks, and their meanings.

dt
    Time step width
i
    Index of a neuron (`NeuronGroup`) or the pre-synaptic neuron
    of a synapse (`Synapses`)
j
    Index of a post-synaptic neuron of a synapse
lastspike
    Last time that the neuron spiked (for refractoriness)
lastupdate
    Time of the last update of synaptic variables in event-driven
    equations (only defined when event-driven equations are used).
N
    Number of neurons (`NeuronGroup`) or synapses (`Synapses`). Use
    ``N_pre`` or ``N_post`` for the number of presynaptic or
    postsynaptic neurons in the context of `Synapses`.
not_refractory
    Boolean variable that is normally true, and false if the neuron
    is currently in a refractory state
t
    Current time
t_in_timesteps
    Current time measured in time steps
xi, xi_*
    Stochastic differential in equations


Event-driven equations
----------------------
Equations defined as event-driven are completely ignored in the state update.
They are only defined as variables that can be externally accessed.
There are additional constraints:

* An event-driven variable cannot be used by any other equation that is not
  also event-driven.
* An event-driven equation cannot depend on a differential equation that is not
  event-driven (directly, or indirectly through subexpressions). It can depend
  on a constant parameter.

Currently, automatic event-driven updates are only possible for one-dimensional
linear equations, but this may be extended in the future.

Equation objects
----------------
The model definitions for `NeuronGroup` and `Synapses` can be simple strings or
`Equations` objects. Such objects can be combined using the add operator::

	eqs = Equations('dx/dt = (y-x)/tau : volt')
	eqs += Equations('dy/dt = -y/tau: volt')

`Equations` allow for the specification of values in the strings, but does this by simple
string replacement, e.g. you can do::
  
  eqs = Equations('dx/dt = x/tau : volt', tau=10*ms)
   
but this is exactly equivalent to::

  eqs = Equations('dx/dt = x/(10*ms) : volt')

The `Equations` object does some basic syntax checking and will raise an error if two equations defining
the same variable are combined. It does not however do unit checking, checking for unknown identifiers or
incorrect flags -- all this will be done during the instantiation of a `NeuronGroup` or `Synapses` object.

Examples of `Equation` objects
------------------------------

**Concatenating equations**

.. doctest::

    >>> membrane_eqs = Equations('dv/dt = -(v + I)/ tau : volt')
    >>> eqs1 = membrane_eqs + Equations('''I = sin(2*pi*freq*t) : volt
    ...                                    freq : Hz''')
    >>> eqs2 = membrane_eqs + Equations('''I : volt''')
    >>> print(eqs1)
    I = sin(2*pi*freq*t) : V
    dv/dt = -(v + I)/ tau : V
    freq : Hz
    >>> print(eqs2)
    dv/dt = -(v + I)/ tau : V
    I : V

**Substituting variable names**

.. doctest::

    >>> general_equation = 'dg/dt = -g / tau : siemens'
    >>> eqs_exc = Equations(general_equation, g='g_e', tau='tau_e')
    >>> eqs_inh = Equations(general_equation, g='g_i', tau='tau_i')
    >>> print(eqs_exc)
    dg_e/dt = -g_e / tau_e : S
    >>> print(eqs_inh)
    dg_i/dt = -g_i / tau_i : S

**Inserting values**

.. doctest::

    >>> eqs = Equations('dv/dt = mu/tau + sigma/tau**.5*xi : volt',
    ...                  mu=-65*mV, sigma=3*mV, tau=10*ms)
    >>> print(eqs)
    dv/dt = (-65. * mvolt)/(10. * msecond) + (3. * mvolt)/(10. * msecond)**.5*xi : V
