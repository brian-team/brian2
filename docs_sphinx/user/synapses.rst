Synapses
========

.. note::
    `Synapses` is now the only class for defining synaptic interactions, it
    replaces *Connections*, *STDP*, etc.

Defining synaptic models
------------------------

The most simple synapse (adding a fixed amount to the target membrane potential
on every spike) is described as follows::

  w = 1*mV
  S = Synapses(P, Q, pre='v += w')

This defines a set of synapses between `NeuronGroup` P and `NeuronGroup` Q.
If the target group is not specified, it is identical to the source group by default.
The ``pre`` keyword defines what happens when a presynaptic spike arrives at
a synapse. In this case, the constant ``w`` is added to variable ``v``.
Because ``v`` is not defined as a synaptic variable, it is assumed by default
that it is a postsynaptic variable, defined in the target `NeuronGroup` Q.
Note that this does not does create synapses (see `Creating Synapses`_), only the
synaptic models.

To define more complex models, models can be described as string equations,
similar to the models specified in `NeuronGroup`::

  S = Synapses(P, Q, model='w : volt', pre='v += w')

The above specifies a parameter ``w``, i.e. a synapse-specific weight.

Synapses can also specify code that should be executed whenever a postsynaptic
spike occurs (keyword ``post``) and a fixed (pre-synaptic) delay for all
synapses (keyword ``delay``). See the reference documentation for `Synapses`
for more details.

Model syntax
^^^^^^^^^^^^
The model follows exactly the same syntax as for `NeuronGroup`. There can be parameters
(e.g. synaptic variable ``w`` above), but there can also be named
subexpressions and differential equations, describing the dynamics of synaptic
variables. In all cases, synaptic variables are created, one value per synapse.
Internally, these are stored as arrays. There are a few things worth noting:

* A variable with the ``_post`` suffix is looked up in the postsynaptic (target) neuron. That is,
  ``v_post`` means variable ``v`` in the postsynaptic neuron.
* A variable with the ``_pre`` suffix is looked up in the presynaptic (source) neuron.
* A variable not defined as a synaptic variable is considered to be postsynaptic.
* A variable not defined as a synaptic variable and not defined in the
  postsynaptic neuron is considered an external constant

For the integration of differential equations, one can use the same keywords as
for `NeuronGroup`.

Event-driven updates
^^^^^^^^^^^^^^^^^^^^
By default, differential equations are integrated in a clock-driven fashion, as for a
`NeuronGroup`. This is potentially very time consuming, because all synapses are updated at every
timestep. It is possible to ask Brian 2 to simulate differential equations in an event-driven fashion
using the keyword ``(event-driven)``. A typical example is pre- and postsynaptic traces in STDP::

  model='''w:1
           dApre/dt=-Apre/taupre : 1 (event-driven)
           dApost/dt=-Apost/taupost : 1 (event-driven)'''

Here, Brian updates the value of ``Apre`` for a given synapse only when this synapse receives a spike,
whether it is presynaptic or postsynaptic. More precisely, the variables are updated every time either
the ``pre`` or ``post`` code is called for the synapse, so that the values are always up to date when
these codes are executed.

Automatic event-driven updates are only possible for a subset of equations, in particular for
one-dimensional linear equations. These equations must also be independent of the other ones,
that is, a differential equation that is not event-driven cannot
depend on an event-driven equation (since the values are not continuously updated).
In other cases, the user can write event-driven code explicitly in the update codes (see below).

Pre and post codes
^^^^^^^^^^^^^^^^^^
The ``pre`` code is executed at each synapse receiving a presynaptic spike. For example::

	pre='v+=w'

adds the value of synaptic variable ``w`` to postsynaptic variable ``v``. As for the model equations,
the ``_post`` (``_pre``) suffix indicates a postsynaptic (presynaptic) variable, and variables not found
in the synaptic variables are considered postsynaptic by default.
Internally, the code is executed for all synapses receiving
presynaptic spikes during the current timestep. Therefore, the code should be understood as acting on
arrays rather than single values. Any sort of code can be executed. For example, the following code defines
stochastic synapses, with a synaptic weight ``w`` and transmission probability ``p``::

	S=Synapses(input,neurons,model="""w : 1
                                      p : 1""",
        	                 pre="v+=w*(rand()<p)")

The code means that ``w`` is added to ``v`` with probability ``p`` (note that, internally, ``rand()``
is transformed to a instruction that outputs an array of random numbers).
The code may also include multiple lines.

As mentioned above, it is possible to write event-driven update code for the synaptic variables.
For this, two special variables are provided: ``t`` is the current time when the code is executed,
and ``lastupdate`` is the last time when the synapse was updated (either through ``pre`` or ``post``
code). An example is short-term plasticity (in fact this could be done automatically with the use
of the ``(event-driven)`` keyword mentioned above)::

	S=Synapses(input,neuron,
	           model='''x : 1
	                    u : 1
	                    w : 1''',
	           pre='''u=U+(u-U)*exp(-(t-lastupdate)/tauf)
	                  x=1+(x-1)*exp(-(t-lastupdate)/taud)
	                  i+=w*u*x
	                  x*=(1-u)
	                  u+=U*(1-u)''')

Summed variables
^^^^^^^^^^^^^^^^
In many cases, the postsynaptic neuron has a variable that represents a sum of variables over all
its synapses. This is called a "summed variable". An example is nonlinear synapses (e.g. NMDA)::

	neurons = NeuronGroup(1, model="""dv/dt=(gtot-v)/(10*ms) : 1
	                                  gtot : 1""")
	S=Synapses(input,neurons,
	           model='''dg/dt=-a*g+b*x*(1-g) : 1
	                    gtot_post = g : 1  (summed)
	                    dx/dt=-c*x : 1
	                    w : 1 # synaptic weight
	                 ''',
	           pre='x+=w')

Here, each synapse has a conductance ``g`` with nonlinear dynamics. The neuron's total conductance
is ``gtot``. The line stating ``gtot_post = g : 1  (summed)`` specifies the link
between the two: ``gtot`` in the postsynaptic group is the summer over all
variables ``g`` of the corresponding synapses. What happens during the
simulation is that at each time step, presynaptic conductances are summed for each neuron and the
result is copied to the variable ``gtot``. Another example is gap junctions::

    neurons = NeuronGroup(N, model='''dv/dt=(v0-v+Igap)/tau : 1
                                      Igap : 1''')
    S=Synapses(neurons,model='''w:1 # gap junction conductance
                                Igap_post = w*(v_pre-v_post): 1 (summed)''')

Here, ``Igap`` is the total gap junction current received by the postsynaptic neuron.

Creating synapses
-----------------
Creating a `Synapses` instance does not create synapses, it only specifies their dynamics.
The following command creates a synapse between neuron ``i`` in the source group and neuron ``j`` in the target group::

    S.connect(i, j)

It is possible to create several synapses for a given pair of neurons::

    S.connect(i, j, n=3)

This is useful for example if one wants to have multiple synapses with different delays.
Multiple synaptic connections can be created in a single statement::

    S.connect(True)
    S.connect([1, 2], [1, 2])
    S.connect(numpy.arange(10), 1)

The first statement connects all neuron pairs.
The second statement creates synapses between neurons 1 and 1, and between neurons 2 and 2.
The third statement creates synapses between the first ten neurons in the source group and neuron 1
in the target group.

One can also create synapses using code::

	S.connect('i==j')
	S.connect('j==((i+1)%N)')

The code is a boolean statement that should return True when a synapse must be created,
where ``i`` is the presynaptic neuron index and ``j`` is the postsynaptic neuron index
(special variables).
Here the first statement creates one-to-one connections, the second statement creates connections
with a ring structure (``N`` is the number of neurons, assumed to defined elsewhere by the user
as an external variable).
This way of creating synapses is generally preferred.

The string expressions can also refer to pre- or postsynaptic variables. This
can be useful for example for spatial connectivity: assuming that the pre- and
postsynaptic groups have parameters ``x`` and ``y``, storing their location, the
following statement connects all cells in a 250 um radius::

    S.connect('sqrt((x_pre-x_post)**2 + (y_pre-y_post)**2) < 250*umeter')

Synapse creation can also be probabilistic by providing a ``p`` argument,
providing the connection probability for each pair of synapses::

    S.connect(True, p=0.1)

This connects all neuron pairs with a probability of 10%. Probabilities can
also be given as expressions, for example to implement a connection probability
that depends on distance::

    S.connect('i != j',
              p='p_max*exp(-(x_pre-x_post)**2+(y_pre-y_post)**2) / (2*(125*umeter)**2)')

If this statement is applied to a `Synapses` object that connects a group to
itself, it prevents self-connections (``i != j``) and connects cells with a
probability that is modulated according to a 2-dimensional Gaussian of the
distance between the cells.

If conditions for connecting neurons are combined with both the ``n`` (number of
synapses to create) and the ``p`` (probability of a synapse) keywords, they are
interpreted in the following way:

    | For every pair i, j:
    |    if condition(i, j) is fulfilled:
    |        Evaluate p(i, j)
    |        If p(i, j) < uniform random number between 0 and 1:
    |            Create n(i, j) synapses for (i, j)


Accessing synaptic variables
----------------------------
Synaptic variables can be accessed in a similar way as `NeuronGroup` variables. They can be indexed
with two indexes, corresponding to the indexes of pre and postsynaptic neurons, and optionally with a third
index in the case of multiple synapses.
Here are a few examples::

    S.w[2, 5] = 1*nS
    S.w[1, :] = 2*nS
    S.w = 1*nS # all synapses assigned
    w0 = S.w[2, 3, 1] # second synapse for connection 2->3
    S.w[2, 3] = (1*nS, 2*nS)
    S.w[group1, group2] = "(1+cos(i-j))*2*nS"
    S.w[:, :] = 'rand()*nS'
    S.w['abs(x_pre-x_post) < 250*umetre'] = 1*nS

Note that it is also possible to index synaptic variables with a single index
(integer, slice, or array), but in this case synaptic indices have to be
provided.

Delays
------
There is a special synaptic variable that is automatically created: ``delay``. It is the propagation delay
from the presynaptic neuron to the synapse, i.e., the presynaptic delay. This
is just a convenience syntax for accessing the delay stored in the presynaptic
pathway: ``pre.delay``. When there is a  postsynaptic code (keyword ``post``),
the delay of the postsynaptic pathway can be accessed as ``post.delay``.

The delay variable(s) can be set and accessed in the same way as other synaptic
variables.

Multiple pathways
-----------------
It is possible to have multiple pathways with different update codes from the same presynaptic neuron group.
This may be interesting in cases when different operations must be applied at different times for the same
presynaptic spike. To do this, specify a dictionary of pathway names and codes::

    pre={'pre_transmission': 'ge+=w',
         'pre_plasticity': '''w=clip(w+Apost,0,inf)
                              Apre+=dApre'''}

This creates two pathways with the given names (in fact, specifying ``pre=code``
is just a shorter syntax for ``pre={'pre': code}``) through which the delay
variables can be accessed.
The following statement, for example, sets the delay of the synapse between the first neurons
of the source and target groups in the ``pre_plasticity`` pathway::

	S.pre_plasticity.delay[0,0] = 3*ms

Monitoring synaptic variables
-----------------------------
A `StateMonitor` object can be used to monitor synaptic variables. For example, the following statement
creates a monitor for variable ``w`` for the synapses 0 and 1::

	M = StateMonitor(S,'w',record=[0,1])

Note that these are *synapse* indices, not neuron indices. More convenient is
to directly index the `Synapses` object, Brian will automatically calculate the
indices for you in this case::

	M = StateMonitor(S,'w',record=S[0, :])  # all synapses originating from neuron 0
	M = StateMonitor(S,'w',record=S['i!=j'])  # all synapses excluding autapses
	M = StateMonitor(S,'w',record=S['w>0'])  # all synapses with non-zero weights (at this time)

The recorded traces can then be accessed in the usual way, again with the
possibility to index the `Synapses` object::

	plot(M.t / ms, M[0].w / nS)  # first synapse
	plot(M.t / ms, M[0, :].w / nS)  # all synapses originating from neuron 0
	plot(M.t / ms, M['w>0'].w / nS)  # all synapses with non-zero weights (at this time)
