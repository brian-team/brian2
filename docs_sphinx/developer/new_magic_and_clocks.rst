New magic and clock behaviour
=============================

Clocks
------

The rule for clocks in Brian 1 was that
you would either specify a clock explicitly, or it would guess it based on the
following rule: if there is no clock defined in the execution frame of the
object being defined, use the default clock; if there is a single clock
defined in that execution frame, use that clock; if there is more than one
clock defined, raise an error. This rule is clearly confusing because, for a
start, it relies on the notion of an execution frame which is a fairly hidden
part of Python, even if it is something similar to the (relatively clearer)
notion of the calling function scope.

The proposed new rule is simply: if the user defines a clock use it, otherwise
use the default clock. This is not quite as flexible as the old rule, but
has the enormous virtue that it makes subtle bugs much more difficult to
introduce.

Incidentally, you could also change the dt of a
clock after it had been defined, which would invalidate any state updaters that
were based on a fixed dt. This is no longer a problem in Brian 2, since state
updaters are re-built at every run so they work fine with a changed dt. It is
important to note that the dt of the respective clock (i.e. in many cases,
``defaultclock.dt``) at the time of the `run` call, not the dt during
the `NeuronGroup` creation, for example, is relevant for the simulation.

Magic
-----

The old rule for MagicNetwork was to gather all instances of each of the various
classes defined in the execution frame that called the run() method (similar to
clocks). Like in the case of clocks, this rule was very complicated to explain
to users and led to some subtle bugs. The most pervasive bug was that if an
object was not deleted, it was still attached to the execution frame and would
be gathered by MagicNetwork. This combined with the fact that there are,
unfortunately, quite a lot of circular references in Brian that cause objects
to often not be deleted. So if the user did something like this::

	def dosim(params):
		...
		run()
		return something
		
	results = []
	for param in params:
		x = dosim(params1)
		results.append(x)
		
Then they would find that the simulation got slower and slower each time,
because the execution frame of the dosim() function is reused for each call,
and so the objects created in the previous run were still there. To fix this
problem users had to do::

	def dosim(params):
		clear(True, True)
		...
		run()
		return something
	...

While this was relatively simple to do, you wouldn't know to do it unless you
were told, so it caused many avoidable bugs.

Another tricky behaviour was that the user might want to do something like this::

	def make_neuron_group(params):
		G = NeuronGroup(...)
		return G
		
	G1 = make_neuron_group(params1)
	G2 = make_neuron_group(params2)
	...
	run()
	
Now G1 and G2 wouldn't be picked up by run() because they were created in the
execution frame of make_neuron_group, not the one that run() was called from.
To fix this, users had to do something like this::

	@magic_return
	def make_neuron_group(params):
		...
		
or::

	def make_neuron_group(params):
		G = NeuronGroup(...)
		magic_register(G, level=1)
		return G
		
Again, reasonably simple but you can't know about them unless you're told.

.. todo:: Describe how this is implemented in Brian2