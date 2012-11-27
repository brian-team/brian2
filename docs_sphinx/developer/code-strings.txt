.. currentmodule:: brian2

Code strings and equations
==========================

The descriptions below are mostly meant to describe the *internal* use of
code strings (expressions and statements) and equations. The user normally
should just give strings as arguments to the respective functions. The user
however can explicitly construct such objects for complete control over the
namespace.

Code strings
------------
A code string is a string that can be evaluated, referencing state variables
(static and differential equations defined in a `NeuronGroup` or `Synapses`
object) and external variables or functions. Such code strings are used in
various places in Brian: 

* Right-hand side of equations
* Reset
* Threshold condition
* Synaptic events triggered by pre- and post-synaptic spikes
* Expressions used to assign to synaptic variables, or create synapses
* Possibly: Expressions to assign to neuron variables

There are two types of code strings: *Expressions* (RHS of equations, threshold
condition, assignments to synaptic variables, creation of synapses) and 
*statements* (Reset, synaptic events), represented by `Expression` and
`Statements` objects (subclasses of `~brian2.equations.codestrings.CodeString`)
respectively.

Such objects save the local and global namespace at the point where they were
created. Optionally, an explicit namespace can be given, either
completely replacing or augmenting the local/global namespace. This information
is not immediately used to resolve identifiers in the code string because state
variables (including for example pre- and postsynaptic state variables in the
context of synaptic events) are not yet known and take precedence over external
variables.

The main purpose of these objects is to encapsulate a combination of a string
and an external namespace, and bundle all the functionality related to
the resolution of external names. The only method that alters the internal
state of the object is `~brian2.equations.codestrings.CodeString.resolve` and
calling this method is a prerequisite for some other methods to work correctly.
Resolving the external variables is not possible without knowing about the
"internal namespace" (state variables and variables with special meaning, e.g.
"i" and "j" in the context of synapses), therefore resolution is not done
automatically at creation time but in the constructor of `NeuronGroup`, etc.
However, in the common usecase of directly passing a string to the
constructor, this distinction is not apparent to the user.  Apart from calls to this
function, the objects are immutable; functions like
`~brian2.equations.codestrings.CodeString.frozen` (replacing the names of
external variables with their values) return a new object instead of
changing its state.

Resolution order
~~~~~~~~~~~~~~~~
Every identifier in the code string is resolved according to the following
resolution order. Note that "internal variables" are not really resolved
(the namespace dictionary of the object does not contain them) but rather
ignored (except for error checking).

.. note::
   [Marcel] Maybe a clearer design would be to actually let the names of
   state variables refer to what are currently
   `~brian2.codegen.specifiers.Specifier` objects in the code generation
   module. Maybe this should be merged with the future framework for memory
   management?

1. "internal variables" -- provided by the class that is the context for the
   code string, e.g. NeuronGroup or Synapses. These internal variables contain
   "special variables" like "t" or "xi" and state variables (including
   automatically generated variable names with _pre and _post suffix in the
   context of synaptic events).
2. explicit namespace provided during the object creation
3. external variables/functions in the local namespace (as saved during the
   object creation)
4. external variables/functions in the global namespace (as saved during the
   object creation)
5. units (the names returned by
   `~brian2.equations.unitcheck.get_default_unit_namespace`, containing all
   registered unit plus the standard units (ms, mV, nS, etc.)
6. Possibly: A standard set of numpy functions (with unit-aware replacements)

An error is raised if an identifier cannot be resolved. If there is more than
one possible resolution, the resolution is performed according to the above
resolution order but a warning of type `ResolutionConflictWarning` is raised.
 
Expressions
~~~~~~~~~~~
The main additional functionality of `Expression` objects is that they
can be evaluated, given values for the "internal namespace". As code generation
is used even for Python code, this evaluation is not used during simulation but
only for tasks like unit checking. Expressions are also converted to a sympy
expression, allowing for symbolic evaluations e.g. determining whether the
expression is linear or separating it into non-stochastic and stochastic
(containing the special variable ``xi``) parts. 

Equations
---------
There are three kinds of equations in Brian: static equations, differential
equations and parameters (which are not really equations, but share some
properties with them). A single equation is encapsulated internally in an
`~brian2.equations.equations.SingleEquation` object, most
importantly containing the name of the variable, the type of the equation, the
expression defining the right-hand side of the equation (empty for parameters),
a unit, and a list of flags.

The `Equations` object encapsulates a list of equations (saved in the private
attribute `Equations._equations`) and allows access to various subsets of it via
properties. In contrast to the Equations object in Brian1 (where it was possible
to concatenate several Equations), an `Equations` object in Brian2 is
self-contained and completely describes the model for a `NeuronGroup` or
`Synapses` object. As it knows about all state variables, it can check for
non-defined identifiers and for correct units. It also runs a number of checks
on the names of all variables. Additional checks can be registered via the
`register_identifier_check` function, allowing for example for the `Synapses`
class to forbid names ending in ``_pre`` or ``_post``. The checking is strict in
the sense that for example a model defining a `NeuronGroup` is not allowed to
define variables ending in ``_pre``, even if this `NeuronGroup` is never used
in the context of `Synapses`. 

However, another basic check cannot performed by the `Equations` class itself: 
Neuronal and synaptic equations allow for different flags. This check is
performed as soon as the `Equations` object is given to the constructor of the
`NeuronGroup` or `Synapses` class or when the constructor constructs
the respective objects from strings.
