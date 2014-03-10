Overview of Brian2
=============================

This page describes some of the concepts in Brian2 and how the different parts
are meant to work together. Of course, nothing here is set in stone at the
current stage...

Magic and clocks
----------------
For a more detailed description, see :doc:`new_magic_and_clocks`.

The clock system has been simplified and is now more explicit: The user either
defines a `Clock` (and passes it for example to the `NeuronGroup`) or the default
clock is used. Changing the dt of a `Clock` after the creation of a
`NeuronGroup` or between runs is no longer a problem.

The "magic" system is also more stringent and easier to explain now: Brian
keeps track of all instances that are created (they also have a unique name,
all this is taken care of in `BrianObject`) -- independent of the execution
frame. All of these objects are used when using the magic run method. There are
also less issues with circular references as for example synapses or monitors
only store *weak references* to their targets and therefore do not keep them
alive on their own. 

Equations
---------
For user-centered documentation see :doc:`../user/equations`, a short developer
document is available as :doc:`equations_namespaces`.
 
`Equations` objects are the core of model descriptions in `NeuronGroup` (and
eventually `Synapses`). Internally, they are implemented as a list of
`~brian2.equations.equations.SingleEquation` objects (which should never be
used by users directly), exposing additional information via properties (e.g.
the names of all state variables, a dictionary of all differential equations,
etc.). `~brian2.equations.equations.SingleEquation` objects are basically a
wrapper around a single line of equations, providing access to the expression,
the unit, the type of equation, etc.

`Equations` do not deal with namespaces, they do not carry more information
than the equation strings (this has the consequence for example concatenating
`Equations` object is possible and does not cause any problems). They are also
immutable, so using the same object in different groups does not lead to any
problems, either. They offer the possibility to specify values or exchange
variable names (this also increases backward-compatiblity), but this does
nothing more than string replacements.

The `Equations` class performs only very generic checks of the equations (e.g.
whether illegal names such as ``t`` are used for state variables, whether the
unit definition is a valid unit, etc.) -- it does not check for consistent
units (because this requires knowledge of the external namespace, or the pre-
and postsynaptic groups in the case of synapses) or correct flags (because they
differ between `NeuronGroup` and `Synapses`, for example).

Variables and namespaces
------------------------
Objects referring to variables and functions, in particular `NeuronGroup`
and `Synapses` provide the context for resolving names in code strings. This
is done via the `Group.resolve` method that returns a `Variable` or `Function`
object for a name. All internal names (state variables defined in the group and
also all variables referenced by this group, e.g. the pre- and postsynaptic
state variables in a `Synapses` object) are stored in the ``variables``
attribute which can be used as a dictionary but is actually a `Variables`
object. Note that `Variable` objects only exist once for every variable, e.g.
the `Synapses` class contains references to the `Variable` objects of the
pre- and postsynaptic classes, not new objects

Groups can also specify a group-specific, explicit namespace that is
stored in their ``namespace`` attribute, used for resolving external names.
For more details on the resolution of names, see :doc:`equations_namespaces`.

State Updaters
--------------
For a more detailed description, see :doc:`state_update`.

State updaters convert equations into abstract code. Any function (or callable,
in general) that is able to convert an `Equations` object into a string of
abstract code can therefore be used as a state updater. Many state updaters
can be described very easily by creating an `ExplicitStateUpdater` object with
a textual description such as ``x_new = x + dt * f(x, t)`` (which should be
understood as :math:`x_{t+dt} = x_t + dt \cdot f(x_t, t)`).

The `StateUpdateMethod` class provides a mechanism for registering new
state updaters, registered state updaters will be considered when no state
updater is explicitly specified. State updaters expose their capabilities via a
``can_integrate`` method, that specifies whether they are able to integrate
the given equations or not (for example, if a state updater does not support
stochastic equations but the equations are stochastic). The order of the
registration also provides an implicit priority: If not state updater is
specified explicitly, the first from the list that claims to be able to
integrate the equations will be chosen.

Code objects and code generation
--------------------------------
The actual computations during a simulation -- the state update, the threshold
testing, the reset code --  are performed by `CodeObject` objects. A group such
as `NeuronGroup` creates the code objects for a certain target language (at
the moment: Python or C++), providing the abstract code, the specifiers and the
namespace to the code generation module. The `CodeObject` objects are executed
every timestep and either directly update state variables (state update, reset)
or return a result of their computation (threshold).
