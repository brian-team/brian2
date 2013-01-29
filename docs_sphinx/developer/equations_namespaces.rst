Equations and namespaces
========================

Equation parsing
----------------

Namespaces
----------
Namespaces (i.e. the link from names used in model equations to the variables/
functions they refer to) are handled via `Namespace` objects internally (users
do not have to use them explicitly, but they can if they want to have
fine-grained control over namespace resolution). All objects that have a
namespace (currently, `NeuronGroup` and `Synapses), save it in a ``_namespace``
attribute and provide a ``namespace`` property to access it (it is not
allowed to be overwritten). The namespace class provides a static convenience
function that returns a `Namespace` instance from a `namespace` argument in a
`NeuronGroup`/`Synapses` constructor (which might be ``None``, a dictionary or
a `Namespace` object) plus (in the case of `Synapses`) a dictionary of suffixes
and referred namespaces.

The `Namespace` object can be used like a dictionary, allowing for getting and
setting (do we need deleting as well?) values. When doing either of those, the
resolution transparently follows the rules described in :doc:`user/equations`,
in particular it takes the referred namespaces into account in the case of the
synapses class.

This is what the relevant part of the `Synapses` constructor looks like::
	...
	# `namespace`, `source` and `target` are the respective constructor arguments
	# 'model` is the `Equations` object defining the model equations
	self._namespace = Namespace.create(namespace,
	                                   model,
	                                   refers={'pre': source.namespace,
	                                           'post': target.namespace,
	                                           '': target.namespace})

Now, for ``S`` being a `Synapses` object, you can refer to ``S.namespace['w']``,
``S.namespace['v_post']``, etc. and get references to the respective variables
of the `Synapses` class or the `NeuronGroup`.

[Question: Could setting linked variable also work using this framework?
This would mean that one would not have to define it as a neuronal parameter...
More or less clear than before? Would avoid copying, wouldn't it?] 

