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
names to `Variable` objects. `Variable` objects contain information *about*
the variable (name, dtype, units) as well as access to the variable's value via
a ``get_value`` method. Some will also allow setting the values via a
corresponding ``set_value`` method. These objects can therefore act as proxies
to the variables' "contents".

Specifiers are used by code generation and for unit checking.

Namespaces
----------
The `namespace` attribute of a group contains information about the external
(variable or function) names used in the equations. It refers to a
`CompoundNamespace` object, containing sub-namespaces. It always contains two
standard namespaces, the namespace containing mathematical functions (see 
`get_default_numpy_namespace`) and a namespace with physical units (see
`DEFAULT_UNIT_NAMESPACE`). In addition, it can contain a "user-defined"
namespace which, if provided, is expected to exhaustively describe the
namespace at the time of a run (i.e. it can be incomplete in the beginning and
then completed later). Writing to the namespace argument of a group
automatically writes to this sub-namespace and an error is raised if it does
not exist.

Objects that have namespaces (`NeuronGroup`, `Synapses`) create this attribute
using the `create_namespace` function, by using such a line in ``__init__``::

	# 'namespace' is the keyword argument, i.e. a dictionary or None
	self.namespace = self.create_namespace(namespace)
	 
An object that does not have a user-defined namespace will fill the namespace
at the time of a run: The `Network.run` function takes an optional
``namespace`` argument that defines the namespace for all objects in the
`Network`, but only for those object that do not have a user-defined namespace.
If this argument is not given, the local context will be used instead.

Internally, this is realized via the ``before_run`` function. At the start of a
run, `Network.before_run` calls `BrianObject.before_run` of every object in the
network with a namespace argument. This namespace argument contains a tuple, 
either ``('implicit-run-namespace', namespace)`` or
``('explicit-run-namespace, namespace)`` (the use for this tuple instead of a
simple dictionary is to have more meaningful warning messages in the case of
a namespace conflict -- but maybe this is an unnecessary complication). When
objects need to resolve identifiers, they use the `CompoundNamespace.resolve`
or `CompountNamespace.resolve_all` methods and pass the given namespace as
an ``additional_namespace`` argument. The methods in `CompoundNamespace`
take care of ignoring it in the case of an existing explicit namespace and
return the value corresponding to the identifier (raising a warning in the case
of an ambiguity or an error if it cannot be resolved). For values that are
passed on to the code generation stage, the units should be removed. The
previously mentioned methods take care of this if the ``strip_units`` argument
is set to ``True``. 
