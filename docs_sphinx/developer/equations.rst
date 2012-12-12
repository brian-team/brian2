Equations
=========

.. note::
   [Marcel] This document contains Romain's notes that are now partly outdated
   but kept here for reference purposes (parts of it should be converted into
   user documentation). For	a documentation of the current Equations system in
   Brian2, see: :doc:`code-strings`.

Equations are used both in NeuronGroup and Synapses to:

* define state variables
* define continuous-updates on these variables, through differential equations

Equations are defined by strings. In Brian 2.0, we completely remove the Equations objects
as they are currently used. Instead, everything is only described by strings. This will
greatly simplify the design, because a lot of the complexity was due to the possibility of
combining equations. But we still keep an ``Equations`` class at least internally,
which handles basic things.

An Equation is a set of single lines in a string:
    (1) ``dx/dt = f : unit`` (differential equation)
    (2) ``x = f : unit`` (static equation)
    (3) ``x : unit`` (parameter)

The equations may be defined on multiple lines with the character \\.
Comments using # may also be included. Subunits are not allowed, i.e., one must
write ``volt``, not ``mV``. But compound units are allowed (e.g.
``farad/meter**2``).

Note that aliases disappear in Brian 2.0.
Some special variables are defined: t, dt (time) and xi (white noise).
Variable names starting with an underscore are forbidden.
But in Brian 2.0, noise is dealt with differently. Equations are stochastic
differential equations, that is, the right hand side is split into two terms:
f(x,t).dt and g(x,t).dW. State updaters can use these two terms separately and
directly. Before, there were only deterministic state updaters, and the stochastic
term was added afterwards.

Flags
-----
Another new syntax if the possibility of flags. A flag is a keyword in brackets, which
qualifies the equations. There are several keywords:

* event-driven: this is only used in Synapses, and means that the differential equation should be updated
  only at the times of events. This implies that the equation is taken out of the continuous
  state update, and instead a event-based state update statement is generated and inserted into
  event codes (pre and post).
  This can only qualify differential equations of synapses. Currently, only one-dimensional
  linear equations can be handled (see below).
* active: this means the variable is only updated during the active (not refractory) state.
  The name may be changed. This can only qualify differential equations of neuron groups.
* constant: this means the parameter will not be changed during a run. This allows
  optimizations in state updaters.
  This can only qualify parameters.

Different flags may be specified as follows::

	dx/dt = f : unit (flag1,flag2)

However, the current flags are mutually exclusive.

Event-driven equations
----------------------
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

Synaptic equations
------------------
A specificity of synaptic equations is that they can include variables of
the pre and postsynaptic groups. The resolution of variable names is as follows:

* A variable with the ``_post`` suffix is looked up in the postsynaptic (target) neuron. That is,
  ``v_post`` means variable ``v`` in the postsynaptic neuron.
* A variable with the ``_pre`` suffix is looked up in the presynaptic (source) neuron.
* A variable not defined as a synaptic variable is considered to be postsynaptic.
* A variable not defined as a synaptic variable and not defined in the postsynaptic neuron is considered
  external.

External variables
------------------
Equations can include variables and functions defined elsewhere. The resolution
order is:

* reserved names (xi, t and dt)
* state variables
* for synapses: pre/post-synaptic variables
* external variables, in the frame nearest that where the string is defined.

We may allow the passing of a namespace for external variables. This could be
done by ``Equations(str,namespace=...)``. This should apply equally to
equations, reset and threshold. When a namespace is specified, external
variables are first looked for in the namespace, then in the external frames.
The difference with Brian 1.0 is that we do not allow the addition of Equations
object anymore, which simplifies the design considerably.

One potential difficulty here is when external functions are used in the equations,
and these functions use external variables not defined in the equations. One
option (currently used in Brian 1) is to use the full namespace where the
string was defined (instead of building the namespace with identified external
variables). Another option, perhaps cleaner, could be to use a decorator::

	@external_variables('tau')
	def f(...):

This would inform Brian to include ``tau`` in the equations namespace, where
comes from the frame where f is defined.

In Brian 1, there are two other types of external variables (rather inputs):
``Timedarray`` and ``Linkedvariable``. We may want to reconsider the way these
are dealt with. Note that lumped variables from synapses are processed with a
``Linkedvariable``.

Processing equations
--------------------
Equations are processed in the following way:

	(1) Parsing. First, we extract individual equations and build the data
	    structure. Stochastic and deterministic parts should be extracted at
	    this stage (using sympy).
	(2) Ordering static variables. There must be no cycle in the dependency
	    graph of static variables. In addition, it is necessary to calculate
	    this graph (which is a set of trees) and compute the update order of
	    these equations. This is required for generating the abstract state
	    update code.
	(3) Variable resolution. Here we identify state and external variables
	    and build a namespace. This step will be done in an external function,
	    so it can also be used for threshold and reset.
	(4) Inspection. This means checking units, linearity, conditional linearity
	    (using sympy).
	    Some or all of it can be done externally (checking units).
	(5) Generating abstract state update code. This will be done in a separate
	    module.
	(6) Generating target-specific state update code. This is going to be in 
		code generation, not in equations objects.

At run time, constant parameters should be frozen. But this could possibly be
done at code generation time (could be better).
After step (5), we may have a series of statements, of the same nature as
reset. One special case is linear updates, which use specific updaters depending
on the target (matrix multiplication).

Step (3) is common to other string-based definitions (threshold, reset).
It needs to be done before inspecting the
equations (e.g. for units checking). Note that units checking could be done in
the same way for expressions (e.g. for threshold).
Therefore, I suggest that the result of parsing is a list of expressions. Steps
(3) and (4) would then be done by the Expression class.

In Brian 1, static variables are included into differential equations, so that
these variables are effectively hidden to the state updaters. This simplifies
the design, but is probably not the most efficient in general. It seems more
efficient to calculate the value of all static variables in a single pass,
at every time step.

We propose to rely more systematically on sympy for inspecting equations (see below).

Parsing
-------
Parsing is now based on pyparsing (MIT license).
Maybe dividing into deterministic and noise terms could be done in this way too?

Units checking
--------------
Currently, to test units, the RHS of each equation is evaluated with the state
variables replaced by units. If it evaluates without error and the units is
consistent with the LHS divided by second, then unit checking succeeds. This
is not ideal because errors can also be raised because of a division by zero,
for example.

Instead, we propose to do it similarly but using sympy. For each SI unit, a symbol
is associated. A namespace is then created, with each state variable and each
external variable mapped to its value times the unit symbol. When then evaluate
RHS-LHS/second and simplify the expression with sympy. The result should be a single
term. Alternatively, we separately evaluate RHS and LHS/second and compare. This
may allow more meaningful error messages.

One potential issue is with external functions. We suggest an optional decorator::

	@has_units(volt)

State updaters
--------------
State updaters generate an abstract state update code from the equations, which
depends on the integration scheme. This code could take the form of a Statements
object.

Currently, in Dan's code, these integration schemes are specified by a generator
function that produces string statements via yield instructions.
It's already better than before, but perhaps it could be improved.
Ideally, one would want to try the scheme as it is written in a math book.
Something like::

	xi(t+dt)=xi(t)+dt*dxi(x(t),t)

(just to give a flavour). However, Dan pointed out that this is not always
possible, for example for the exponential Euler scheme. However, I feel that
all explicit schemes could be written in this way, and allowing such a syntax
would make it very easy for one to write state updaters, including with
stochastic updaters.

Given that we are going to have special cases (linear updater), it might be
ok to have special cases for implicit updaters. Something to think about.
Should we write down the maths for a few state updaters?

One new element of design in Brian 2.0 is that noise is managed directly by
state updaters. Another potentially new element is that static functions may
not be included in the RHS of differential equations, but rather in the update
code. This could be done automatically, I think.
