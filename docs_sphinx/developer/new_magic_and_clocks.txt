New magic and clock behaviour
=============================

Dan and Marcel came up with the following proposal for how the magic and clocks
should function in Brian 2.0.

Some background: both magic and clocks had a problem in Brian 1.x that they
were incredibly complicated to explain, confusing for the user, and that this
led to many subtle bugs or poorly performing code.

Clocks
------

The rule for clocks was that
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
were based on a fixed dt. Consequently, in the new design you cannot change the
dt of a clock once it has been specified. This has one huge problem, it would
mean that you couldn't change the dt of the defaultclock, which would be too
annoying in practice. So, you can now leave the dt of a clock unspecified when
you create the clock, and it can be left unspecified until the .dt attribute
is accessed. If no value is specified, it uses ``dt=0.1*ms``, and the .dt
attribute can be set precisely once (if it was left unspecified when created).
This means you can now set ``defaultclock.dt`` precisely once. Again, this is
slightly less flexible than the old system, but avoids many potentially
confusing bugs.

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

Therefore, the suggestion for the new system is to make magic functions gather
all instances unless they have explicitly been excluded via forget(), or
possibly by a new .deactivate() or .forget() method. The first type of bug
still remains, but when we combine this with the suggestions in the
defensive programming document, and make sure that the run() function does a
garbage collection before gathering instances, we should be able to eliminate
the circular references in most cases. In the remaining cases, the problems
should hopefully be just that the simulation runs slower, and not that it gives
incorrect results. To ensure this, we should write automated tests to check that
circular references are avoided, e.g. by doing something like::

	import gc
	for i in xrange(100):
		obj = SomeBrianClass(...)
	gc.collect()
	objs = get_instances(SomeBrianClass)
	assert_equal(len(objs), 0)
	
To make circular references easier to avoid, Brian objects should use
weak references to other Brian objects rather than direct references. This has
two benefits: (1) it eliminates circular references (because weakrefs don't
count towards this), (2) it helps to avoid certain types of bugs described in
the defensive programming document.