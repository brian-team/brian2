Equations
=========

.. _equation_strings:

Equation strings
----------------
Equations are used both in `NeuronGroup` and `Synapses` to:

* define state variables
* define continuous-updates on these variables, through differential equations

Equations are defined by multiline strings.

An Equation is a set of single lines in a string:
    (1) ``dx/dt = f : unit`` (differential equation)
    (2) ``x = f : unit`` (subexpression)
    (3) ``x : unit`` (parameter)

The equations may be defined on multiple lines (no explicit line continuation with ``\`` is necessary).
Comments using ``#`` may also be included. Subunits are not allowed, i.e., one must write ``volt``, not ``mV``. This is
to make it clear that the values are internally always saved in the basic units, so no confusion can arise when getting
the values out of a `NeuronGroup` and discarding the units. Compound units are of course allowed as well (e.g. ``farad/meter**2``).
There are also three special "units" that can be used: ``1`` denotes a dimensionless floating point variable,
``boolean`` and ``integer`` denote dimensionless variables of the respective kind.

Some special variables are defined: `t`, `dt` (time) and `xi` (white noise).
Variable names starting with an underscore and a couple of other names that have special meanings under certain
circumstances (e.g. names ending in ``_pre`` or ``_post``) are forbidden.

For stochastic equations with several ``xi`` values it is now necessary to make clear whether they correspond to the same
or different noise instantiations. To make this distinction, an arbitrary suffix can be used, e.g. using ``xi_1`` several times
refers to the same variable, ``xi_2`` (or ``xi_inh``, ``xi_alpha``, etc.) refers to another. An error will be raised if
you use more than one plain ``xi``. Note that noise is always independent across neurons, you can only work around this
restriction by defining your noise variable as a shared parameter and update it using a user-defined function (e.g. with `~Group.run_regularly`),
or create a group that models the noise and link to its variable (see :ref:`linked_variables`).

Flags
~~~~~
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
*shared*
  this means that a parameter or subexpression is not neuron-/synapse-specific
  but rather a single value for the whole `NeuronGroup` or `Synapses`. A shared
  subexpression can only refer to other shared variables.
*linked*
  this means that a parameter refers to a parameter in another `NeuronGroup`.
  See :ref:`linked_variables` for more details.

Multiple flags may be specified as follows::

	dx/dt = f : unit (flag1,flag2)

Event-driven equations
~~~~~~~~~~~~~~~~~~~~~~
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
~~~~~~~~~~~~~~~~
The model definitions for `NeuronGroup` and `Synapses` can be simple strings or
`Equations` objects. Such objects can be combined using the add operator::

	eqs = Equations('dx/dt = (y-x)/tau : volt')
	eqs += Equations('dy/dt = -y/tau: volt')

`Equations` allow for the specification of values in the strings, but do this by simple
string replacement, e.g. you can do::
  
  eqs = Equations('dx/dt = x/tau : volt', tau=10*ms)
   
but this is exactly equivalent to::

  eqs = Equations('dx/dt = x/(10*ms) : volt')

The `Equations` object does some basic syntax checking and will raise an error if two equations defining
the same variable are combined. It does not however do unit checking, checking for unknown identifiers or
incorrect flags -- all this will be done during the instantiation of a `NeuronGroup` or `Synapses` object.


.. _external-variables:

External variables and functions
--------------------------------
Equations defining neuronal or synaptic equations can contain references to
external parameters or functions. These references are looked up at the time
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

Examples
--------

Equation objects
~~~~~~~~~~~~~~~~
**Concatenating equations**

.. doctest::

	>>> membrane_eqs = Equations('dv/dt = -(v + I)/ tau : volt')
	>>> eqs1 = membrane_eqs + Equations('''I = sin(2*pi*freq*t) : volt
	...                                    freq : Hz''')
	>>> eqs2 = membrane_eqs + Equations('''I : volt''')
	>>> print eqs1
	I = sin(2*pi*freq*t)  : V
	dv/dt = -(v + I)/ tau  : V
	freq : Hz
	>>> print eqs2
	dv/dt = -(v + I)/ tau  : V
	I : V

**Substituting variable names**

.. doctest::

	>>> general_equation = 'dg/dt = -g / tau : siemens'
	>>> eqs_exc = Equations(general_equation, g='g_e', tau='tau_e')
	>>> eqs_inh = Equations(general_equation, g='g_i', tau='tau_i')
	>>> print eqs_exc
	dg_e/dt = -g_e / tau_e  : S
	>>> print eqs_inh
	dg_i/dt = -g_i / tau_i  : S

**Inserting values**

.. doctest::

	>>> eqs = Equations('dv/dt = mu/tau + sigma/tau**.5*xi : volt',
	                    mu = -65*mV, sigma=3*mV, tau=10*ms)
	>>> print eqs
	dv/dt = (-0.065 * volt)/(10.0 * msecond) + (3.0 * mvolt)/(10.0 * msecond)**.5*xi  : V

