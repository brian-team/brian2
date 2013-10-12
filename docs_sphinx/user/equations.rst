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
    (2) ``x = f : unit`` (static equation)
    (3) ``x : unit`` (parameter)

The equations may be defined on multiple lines (no explicit line continuation with ``\`` is necessary).
Comments using ``#`` may also be included. Subunits are not allowed, i.e., one must write ``volt``, not ``mV``. This is
to make it clear that the values are internally always saved in the basic units, so no confusion can arise when getting
the values out of a `NeuronGroup` and discarding the units. Compound units are of course allowed as well (e.g. ``farad/meter**2``).

Aliases are no longer available in Brian 2. Some special variables are defined: `t`, `dt` (time) and `xi` (white noise).
Variable names starting with an underscore and a couple of other names that have special meanings under certain
circumstances (e.g. names ending in ``_pre`` or ``_post``) are forbidden.

For stochastic equations with several ``xi`` values it is now necessary to make clear whether they correspond to the same
or different noise instantiations. Two make this distinction, an arbitrary suffix can be used, e.g. using ``xi_1`` several times
refers to the same variable, ``xi_2`` (or ``xi_inh``, ``xi_alpha``, etc.) refers to another. An error will be raised if
you use more than one plain ``xi``. Note that noise is always independent across neurons, you can only work around this
restriction by defining your noise variable as a parameter and update it using a user-defined function. 

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
  optimizations in state updaters.
  This can only qualify parameters.

Different flags may be specified as follows::

	dx/dt = f : unit (flag1,flag2)

However, the current flags are mutually exclusive.

Event-driven equations
~~~~~~~~~~~~~~~~~~~~~~
Equations defined as event-driven are completely ignored in the state update.
They are only defined as variables that can be externally accessed.
There are additional constraints:

* An event-driven variable cannot be used by any other equation that is not
  also event-driven.
* An event-driven equation cannot depend on a differential equation that is not
  event-driven (directly, or indirectly through static equations). It can depend
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


External variables and functions
--------------------------------
Equations defining neuronal or synaptic equations can contain references to
external parameters or functions. During the initialisation of a `NeuronGroup`
or a `Synapses` object, this *namespace* can be provided as an argument. In
this case, it is assumed that this namespace is exhaustive and contains all
external identifiers (note that units and a set of standard numpy functions
are always provided and should not be included in this namespace). This
namespace does not necessarily need to be exhaustive at the time of the creation
of the `NeuronGroup`/`Synapses`, entries can be added (or modified) at a later
stage via the `namespace` attribute (e.g. ``G.namespace['tau'] = 10*ms``) -- it
only has to be complete at the time of the run.

If no such namespace is provided, the namespace will be filled at the point of
the call to the `run` function. Either via an explicit namespace argument to
the `run` function or -- if this is  not provided -- from the variables
defined at that point in the code (more specifically, the variables from the
*locals* and *globals* symbol table). This namespace is shared among all
objects in the network that do not have their own explicit namespace.

To summarize: The namespace of external identifiers for an object such as a
`NeuronGroup` or `Synapses` is:

* The object's explicit namespace if it is provided at creation time.
* If no explicit namespace is given, the namespace argument of the run
  function is used.
* If neither the object, nor the run function received a namespace argument,
  the variables from the context of the run function are used.

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

Resolution order
~~~~~~~~~~~~~~~~
For each identifier (variable or function name) in the model equations, a corresponding object will be
determined using the resolution order specified below. If not resolution can be found, an error will be raised.
If more than one resolution is possible, the first in the resolution order will be used but a warning will be
raised.

1. "special variables": `t`, `dt`, `xi` (and `xi_...`)
2. state variables of the `NeuronGroup`/`Synapses` itself.
3. variables from "referred namespaces", i.e. in the `Synapses` class, variables
   from the pre-synaptic group (using a ``_pre`` suffix) or from the post-synaptic
   group (using a ``_post`` suffix or no suffix).
4. A standard set of numpy functions (with unit-aware/code-generation
   replacements, the names in
   `~brian2.core.namespace.get_default_numpy_namespace`).
5. units (the names in `~brian2.core.namespace.DEFAULT_UNIT_NAMESPACE`),
   containing all registered units plus the standard units (ms, mV, nS, etc.)
6. Explicitly given entries in the namespace dictionary of the object,
   explicitly given entries to the `run` function or variables from the local
   context (see explanations in the previous section)

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

