Equations and namespaces
========================

Equation parsing
----------------

Namespaces
----------
Namespaces (i.e. the link from names used in model equations to the variables/
functions they refer to) are handled via `ModelNamespace` objects internally.
The `ModelNamespace` object represents a "compound namespace", containing
several subnamespaces (e.g. the model variables, the user-specified namespace,
the local variables, the global variables, etc.)
All objects that have a namespace (currently, `NeuronGroup` and `Synapses`),
inherit from `ObjectWithNamespace` [maybe we could find a better name? Should
we merge the class with `BrianObject` maybe?] and save the local and global
namespace on creation. They are then supposed to initialize the `_namespace`
attribute with a new `ModelNamespace` object. Typically, this should be done
using the `create_namespace` method provided by `ObjectWithNamespace`. In a
`NeuronGroup` this means having the following line in `__init__`::

	# 'namespace' is the keyword argument, i.e. a dictionary or None
	self._namespace = self.create_namespace(self.specifiers, namespace) 

In `Synapses`, the namespace also refers the model namespace of the source and
the target group::

	# 'source', 'target' and 'namespace' are the respective arguments
	self._namespace = self.create_namespace(self.specifiers, namespace,
	                                        Namespace('presynaptic',
	                                                  source.namespace,
	                                                  suffixes=['_pre']),
	                                        Namespace('postsynaptic',
	                                                  target.namespace,
	                                                  suffixes=['_post', '']))

The `namespace` attribute of a `NeuronGroup`/`Synapses` can be used like a
dictionary, allowing for getting and setting values. Setting values will set
values only in the 'user-defined' subnamespace, all other namespaces will never
be changed.

Now, for ``S`` being a `Synapses` object, you can refer to ``S.namespace['w']``,
``S.namespace['v_post']``, etc. and get references to the respective variables
of the `Synapses` class or the `NeuronGroup`.

[Question: Could setting linked variable also work using this framework?
This would mean that one would not have to define it as a neuronal parameter...
More or less clear than before? Would avoid copying, wouldn't it?] 

