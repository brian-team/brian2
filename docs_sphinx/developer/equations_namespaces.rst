Equations and namespaces
========================

Equation parsing
----------------
Parsing is done via `pyparsing`_, for now find the grammar at the top of the
`brian2.equations.equations` file.

.. _pyparsing: http://pyparsing.wikispaces.com/

Specifiers
----------
.. note : The names and the object hierarchy will probably change in the future.

Each Brian object that saves state variables (e.g. `NeuronGroup`, `Synapses`,
`StateMonitor`) has a ``specifiers`` attribute, a dictionary mapping variable
names to `Specifier` objects. `Specifier` objects contain information *about*
the variable (name, dtype, units) as well as access to the variable's value via
a ``get_value`` method. Some will also allow setting the values via a
corresponding ``set_value`` method. These objects can therefore act as proxies
to the variables' "contents".

Specifiers are used by code generation and for unit checking. In the case of
synapses, the `Synapses` object itself should only save the specifiers for its
own state variables (e.g. the synaptic weights). When passing a specifiers
dictionary to code generation or unit checking, it needs however to pass a
dictionary not only containing its own specifiers but also the specifiers of
the pre- and postsynaptic group (with corresponding suffixes for the keys). It
is important that it does not store this "extended" dictionary in its own
specifiers attribute as that would break constructions such as a synapse that
is connected to another synapse (we don't want an expression like
``w_post_pre`` to refer to ``w`` of the synapse itself...).

Namespaces
----------
The `namespace` attribute of a group contains information about the external
(variable or function) names used in the equations. It refers to a
`CompoundNamespace` object, containing sub-namespaces (e.g. the user-specified
namespace, the local variables, the global variables, the units, etc.)
All objects that have a namespace (currently, `NeuronGroup` and `Synapses`),
inherit from `ObjectWithNamespace` [maybe we could find a better name?] and
save the local and global namespace on creation. They are then supposed to
initialize the `namespace` attribute with a new `CompoundNamespace` object.
Typically, this should be done using the `create_namespace` method provided by
`ObjectWithNamespace`. This means having the following line in `__init__`::

	# 'namespace' is the keyword argument, i.e. a dictionary or None
	self.namespace = self.create_namespace(self.N, namespace) 

The `namespace` attribute of a `NeuronGroup`/`Synapses` can be used like a
read-only dictionary. The `~ObjectWithNamespace.resolve_all` method can be used
to convert it to a standard dictionary, containing only the given identifiers.
This method will raise errors if an identifier cannot be resolved and warnings
for ambiguous identifiers. 