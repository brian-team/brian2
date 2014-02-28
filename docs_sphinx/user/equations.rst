Equations
=========

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

Aliases are no longer available in Brian 2. Some special variables are defined: `t`, `dt` (time) and `xi` (white noise).
Variable names starting with an underscore and a couple of other names that have special meanings under certain
circumstances (e.g. names ending in ``_pre`` or ``_post``) are forbidden.

For stochastic equations with several ``xi`` values it is now necessary to make clear whether they correspond to the same
or different noise instantiations. To make this distinction, an arbitrary suffix can be used, e.g. using ``xi_1`` several times
refers to the same variable, ``xi_2`` (or ``xi_inh``, ``xi_alpha``, etc.) refers to another. An error will be raised if
you use more than one plain ``xi``. Note that noise is always independent across neurons, you can only work around this
restriction by defining your noise variable as a scalar parameter and update it using a user-defined function (e.g. a `CodeRunner`).

Flags
~~~~~
A new syntax is the possibility of *flags*. A flag is a keyword in brackets, which
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
*scalar*
  this means that a parameter or subexpression isn't neuron-/synapse-specific
  but rather a single value for the whole `NeuronGroup` or `Synapses`. A scalar
  subexpression can only refer to other scalar variables.

Different flags may be specified as follows::

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
  on a constant parameter. An open question is whether we should also allow it
  to depend on a parameter not defined as constant (I would say no).

Currently, automatic event-driven updates are only possible for one-dimensional
linear equations, but it could be extended.

Equation objects
~~~~~~~~~~~~~~~~
The model definitions for `NeuronGroup` and `Synapses` can be simple strings or
`Equations` objects. Such objects can be combined using the add operator::

	eqs = Equations('dx/dt = (y-x)/tau : volt')
	eqs += Equations('dy/dt = -y/tau: volt')

In contrast to Brian 1, `Equations` objects do not save the surrounding namespace (which led to a lot
of complications when combining equations), they are mostly convenience wrappers
around strings. They do allow for the specification of values in the strings, but do this by simple
string replacement, e.g. you can do::
  
  eqs = Equations('dx/dt = x/tau : volt', tau=10*ms)
   
but this is exactly equivalent to::

  eqs = Equations('dx/dt = x/(10*ms) : volt')

In contrast to Brian 1, specifying the value of a variable using a keyword argument does not mean you
have to specify the values for all external variables by keywords.
[Question: Useful to have the same kind of classes for Thresholds and Resets (Expression and Statements) just
for convenience?]

The `Equations` object does some basic syntax checking and will raise an error if two equations defining
the same variable are combined. It does not however do unit checking, checking for unknown identifiers or
incorrect flags -- all this will be done during the instantiation of a `NeuronGroup` or `Synapses` object.


.. _external-variables:

External variables and functions
--------------------------------
Equations defining neuronal or synaptic equations can contain references to
external parameters or functions. During the initialisation of a `NeuronGroup`
or a `Synapses` object, this *namespace* can be provided as an argument. This
is a group-specific namespace that will only be used for names in the context
of the respective group. Note that units and a set of standard functions are
always provided and should not be given explicitly.
This namespace does not necessarily need to be exhaustive at the time of the
creation of the `NeuronGroup`/`Synapses`, entries can be added (or modified)
at a later stage via the `namespace` attribute (e.g.
``G.namespace['tau'] = 10*ms``).

At the point of the call to the `Network.run` namespace, any group-specific
namespace will be augmented by the "run namespace". This namespace can be
either given explicitly as an argument to the `~Network.run` method or it will
be taken from the locals and globals surrounding the call. A warning will be
emitted if a name is defined in more than one namespace.

To summarize: an external identifier will be looked up in the context of an
object such as `NeuronGroup` or `Synapses`. It will follow the following
resolution hierarchy:

1. Default unit and function names.
2. Names defined in the explicit group-specific namespace.
3. Names in the run namespace which is either explicitly given or the implicit
   namespace surrounding the run call.

Note that if you completely specify your namespaces at the `Group` level, you
should probably pass an empty dictionary as the namespace argument to the
`~Network.run` call -- this will completely switch off the "implicit namespace"
mechanism.

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

External variables are free to change between runs (but not during one run),
the value at the time of the `run` call is used in the simulation. 

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

