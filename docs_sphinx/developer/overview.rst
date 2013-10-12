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
-------------------------
Objects referring to variables and functions, in particular `NeuronGroup`
and `Synapses` have two dictionary-like attributes: ``namespace``
and ``variables``. The *namespace* is related to everything external to the
model itself, i.e. variables and functions defined outside of the model
equations. It by default consists of a set of standard units and functions 
and optionally of an explicitly given namespace. If no namespace is given
explicitly, the namespace used for running code will be filled at the time of 
a run with either the namespace provided to the run function or the
locals/globals surrounding the run call. *Variables* on the other hand,
define everything *internal* to the model, the objects in this dictionary
inherit from `Variable` and -- in addition to specifying things like the units
-- act as proxy objects, connecting for example state variable names to the
numpy arrays where the values are actually stored.

This indirection will be useful when dealing with memory on devices. The
variables also offer an explicit and simple way to implement linked variables
or the access to pre- and postsynaptic variables in `Synapses`: To link the
symbol ``v_post`` to the postsynaptic membrane potentials, the specifier
dictionary just has to contain a reference to the respective `Variable` object
of the target group under the key ``v_post``.

Another parent class of `NeuronGroup` is `Group`, which also relies on the
`Variable` objects and exposes access to the state variables as attributes.

`Variable` objects only exist once for every variable, e.g. the `Synapses`
class contains references to the `Variable` objects of the pre- and postsynaptic
classes.

Indices
-------

To handle arrays in generated code correctly, information about *indexing* has
to be stored as well. Every `Group` has two attributes responsible for that:
``indices``, a mapping from index names to index objects and ``variable_indices``,
a mapping from variable names to index names. An index object needs to have
a length and be indexable -- e.g. it could be a numpy array.

For simple classes such as `NeuronGroup` that only have a single index for all
variables (``'_idx'``), nothing needs to be done. ``Group`` automatically
creates an ``indices`` attribute mapping ``'_idx'`` to the ``item_mapping``
and a ``variable_indices`` attribute that maps everything to ``'_idx'``.

TODO: Explain ``item_mapping``


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
