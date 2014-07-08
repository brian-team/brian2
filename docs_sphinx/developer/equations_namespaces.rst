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
called.

Internally, this is realized via the ``before_run`` function. At the start of a
run, `Network.before_run` calls `BrianObject.before_run` of every object in the
network with a namespace argument and a level. If the namespace argument is
given (even if it is an empty dictionary), it will be used together with any
group-specific namespaces for resolving names. If it is not specified or
``None``, the given level will be used to go up in the call frame and determine
the respective locals and globals.