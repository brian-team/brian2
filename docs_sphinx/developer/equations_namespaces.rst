Equations and namespaces
========================

Equation parsing
----------------
Parsing is done via `pyparsing`_, for now find the grammar at the top of the
`brian2.equations.equations` file.

.. _pyparsing: http://pyparsing.wikispaces.com/

Variables
----------
Each Brian object that saves state variables (e.g. `NeuronGroup`, `Synapses`,
`StateMonitor`) has a ``variables`` attribute, a dictionary mapping variable
names to `Variable` objects (in fact a `Variables` object, not a simple
dictionary). `Variable` objects contain information *about*
the variable (name, dtype, units) as well as access to the variable's value via
a ``get_value`` method. Some will also allow setting the values via a
corresponding ``set_value`` method. These objects can therefore act as proxies
to the variables' "contents".

`Variable` objects provide the "abstract namespace" corresponding to a chunk
of "abstract code", they are all that is needed to check for syntactic
correctness, unit consistency, etc.

Namespaces
----------
The `namespace` attribute of a group can contain information about the external
(variable or function) names used in the equations. It specifies a
group-specific namespace used for resolving names in that group. At run time,
this namespace is combined with a "run namespace". This namespace is either
explicitly provided to the `Network.run` method, or the implicit namespace
consisting of the locals and globals around the point where the run function is
called is used. This namespace is then passed down to all the objects via
`Network.before_fun` which calls all the individual `BrianObject.before_run`
methods with this namespace.
