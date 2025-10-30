Namespaces
==========

`Equations` can contain references to
external parameters or functions. During the initialisation of a `NeuronGroup`
or a `Synapses` object, this *namespace* can be provided as an argument. This
is a group-specific namespace that will only be used for names in the context
of the respective group. Note that units and a set of standard functions are
always provided and should not be given explicitly.
This namespace does not necessarily need to be exhaustive at the time of the
creation of the `NeuronGroup`/`Synapses`, entries can be added (or modified)
at a later stage via the `namespace` attribute (e.g.
``G.namespace['tau'] = 10*ms``).

At the point of the call to the `Network.run` namespace, any group-specific
namespace will be augmented by the "run namespace". This namespace can be
either given explicitly as an argument to the `~Network.run` method or it will
be taken from the locals and globals surrounding the call. A warning will be
emitted if a name is defined in more than one namespace.

To summarize: an external identifier will be looked up in the context of an
object such as `NeuronGroup` or `Synapses`. It will follow the following
resolution hierarchy:

1. Default unit and function names.
2. Names defined in the explicit group-specific namespace.
3. Names in the run namespace which is either explicitly given or the implicit
   namespace surrounding the run call.

Note that if you completely specify your namespaces at the `Group` level, you
should probably pass an empty dictionary as the namespace argument to the
`~Network.run` call -- this will completely switch off the "implicit namespace"
mechanism.

The following three examples show the different ways of providing external
variable values, all having the same effect in this case::

	# Explicit argument to the NeuronGroup
	G = NeuronGroup(1, 'dv/dt = -v / tau : 1', namespace={'tau': 10*ms})
	net = Network(G)
	net.run(10*ms)

	# Explicit argument to the run function
	G = NeuronGroup(1, 'dv/dt = -v / tau : 1')
	net = Network(G)
	net.run(10*ms, namespace={'tau': 10*ms})

	# Implicit namespace from the context
	G = NeuronGroup(1, 'dv/dt = -v / tau : 1')
	net = Network(G)
	tau = 10*ms
	net.run(10*ms)

External variables are free to change between runs (but not during one run),
the value at the time of the `run` call is used in the simulation.
