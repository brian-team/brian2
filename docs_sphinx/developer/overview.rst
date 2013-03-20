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
clock is used. You can only set the ``dt`` of a clock once (there were all kinds
of issues in Brian1 if you changed the ``dt`` of a clock after you created a
group, for example).  

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

Specifiers and namespaces
-------------------------
The `ObjectWithNamespace` (suggestions for a better name are welcome...) class
is a parent class for objects referring to variables and functions, in particular
`NeuronGroup` (and eventually `Synapses`). Such an object has two dictionary-like
attributes: ``namespace`` and ``specifiers``. A *namespace* is everything
external to the model itself, e.g. variables or functions defined outside of the
model equations. The namespace by default consists of a set of standard units
and functions + either an explicitly given namespace (for full control) or the
locals and globals if no namespace is given. *Specifiers* on the other hand,
define everything *internal* to the model, the objects in this dictionary
inherit from `Specifier` and -- in addition to specifying things like the units
-- act as proxy objects, connecting for example state variable names to the
numpy arrays where the values are actually stored.

This indirection will be useful when dealing with memory on devices. The
specifiers also offer an explicit and simple way to implement linked variables
or the access to pre- and postsynaptic variables in `Synapses`: To link the
symbol ``v_post`` to the postsynaptic membrane potentials, the specifier
dictionary just has to contain a reference to the respective `Specifier` object
of the target group under the key ``v_post``.

Another parent class of `NeuronGroup` is `Group`, which also relies on the
`Specifier` objects and exposes access to the state variables as attributes.
This is also used in classes such as `StateMonitor` (which is also the reason
why this is not merged with `ObjectWithNamespace` -- `StateMonitor` objects do
not deal with namespaces).

State Updaters
--------------
For a more detailed description, see :doc:`state_update`.

State updaters convert equations into abstract code. Any function (or callable,
in general) that is able to convert an `Equations` object into a string of
abstract code can therefore be used as a state updater. Many state updaters
can be described very easily by creating an `ExplicitStateUpdater` object with
a textual description such as ``return x + dt * f(x, t)`` (which should be
understood as :math:`x_{t+1} = x_t + dt \cdot f(x_t, t)`).

The `StateUpdateMethod` class provides a mechanism for registering new
state updaters, registered state updaters will be considered when no state
updater is explicitly specified. State updaters expose their capabilities via a
``get_priority`` method, that specifies whether they are unable to integrate
the given equations (by returning a priority of 0; for example, if a state
updater does not support stochastic equations but the equations are stochastic)
or what their priority is so that one state updater can be chosen when several
state updaters are able to do the job. 

Code objects and code generation
--------------------------------
For a more detailed description (possibly partly outdated), see
:doc:`codegen/index`.

The actual computations during a simulation -- the state update, the threshold
testing, the reset code --  are performed by `CodeObject` objects. A group such
as `NeuronGroup` creates the code objects for a certain target language (at
the moment: Python or C++), providing the abstract code, the specifiers and the
namespace to the code generation module. The `CodeObject` objects are executed
every timestep and either directly update state variables (state update, reset)
or return a result of their computation (threshold).
