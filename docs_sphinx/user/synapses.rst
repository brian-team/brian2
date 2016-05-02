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
  S = Synapses(P, Q, on_pre='v += w')

This defines a set of synapses between `NeuronGroup` P and `NeuronGroup` Q.
If the target group is not specified, it is identical to the source group by default.
The ``on_pre`` keyword defines what happens when a presynaptic spike arrives at
a synapse. In this case, the constant ``w`` is added to variable ``v``.
Because ``v`` is not defined as a synaptic variable, it is assumed by default
that it is a postsynaptic variable, defined in the target `NeuronGroup` Q.
Note that this does not does create synapses (see `Creating Synapses`_), only the
synaptic models.

To define more complex models, models can be described as string equations,
similar to the models specified in `NeuronGroup`::

  S = Synapses(P, Q, model='w : volt', on_pre='v += w')

The above specifies a parameter ``w``, i.e. a synapse-specific weight.

Synapses can also specify code that should be executed whenever a postsynaptic
spike occurs (keyword ``on_post``) and a fixed (pre-synaptic) delay for all
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
timestep and Brian will therefore emit a warning. If you are sure about integrating the equations at
every timestep (e.g. because you want to record the values continuously), then you should specify
the flag ``(clock-driven)``. To ask Brian 2 to simulate differential equations in an event-driven fashion
use the flag ``(event-driven)``. A typical example is pre- and postsynaptic traces in STDP::

  model='''w:1
           dApre/dt=-Apre/taupre : 1 (event-driven)
           dApost/dt=-Apost/taupost : 1 (event-driven)'''

Here, Brian updates the value of ``Apre`` for a given synapse only when this synapse receives a spike,
whether it is presynaptic or postsynaptic. More precisely, the variables are updated every time either
the ``on_pre`` or ``on_post`` code is called for the synapse, so that the values are always up to date when
these codes are executed.

Automatic event-driven updates are only possible for a subset of equations, in particular for
one-dimensional linear equations. These equations must also be independent of the other ones,
that is, a differential equation that is not event-driven cannot
depend on an event-driven equation (since the values are not continuously updated).
In other cases, the user can write event-driven code explicitly in the update codes (see below).

Pre and post codes
^^^^^^^^^^^^^^^^^^
The ``on_pre`` code is executed at each synapse receiving a presynaptic spike. For example::

	on_pre='v+=w'

adds the value of synaptic variable ``w`` to postsynaptic variable ``v``. As for the model equations,
the ``_post`` (``_pre``) suffix indicates a postsynaptic (presynaptic) variable, and variables not found
in the synaptic variables are considered postsynaptic by default.
Internally, the code is executed for all synapses receiving
presynaptic spikes during the current timestep. Therefore, the code should be understood as acting on
arrays rather than single values. Any sort of code can be executed. For example, the following code defines
stochastic synapses, with a synaptic weight ``w`` and transmission probability ``p``::

	S=Synapses(input,neurons,model="""w : 1
                                      p : 1""",
        	                 on_pre="v+=w*(rand()<p)")

The code means that ``w`` is added to ``v`` with probability ``p`` (note that, internally, ``rand()``
is transformed to a instruction that outputs an array of random numbers).
The code may also include multiple lines.

As mentioned above, it is possible to write event-driven update code for the synaptic variables.
For this, two special variables are provided: ``t`` is the current time when the code is executed,
and ``lastupdate`` is the last time when the synapse was updated (either through ``on_pre`` or ``on_post``
code). An example is short-term plasticity (in fact this could be done automatically with the use
of the ``(event-driven)`` keyword mentioned above)::

	S=Synapses(input,neuron,
	           model='''x : 1
	                    u : 1
	                    w : 1''',
	           on_pre='''u=U+(u-U)*exp(-(t-lastupdate)/tauf)
	                  x=1+(x-1)*exp(-(t-lastupdate)/taud)
	                  i+=w*u*x
	                  x*=(1-u)
	                  u+=U*(1-u)''')

By default, the ``pre`` pathway is executed before the ``post`` pathway (both
are executed in the ``'synapses'`` scheduling slot, but the ``pre`` pathway has
the ``order`` attribute -1, wheras the ``post`` pathway has ``order`` 1. See
:ref:`scheduling` for more details).

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
	           on_pre='x+=w')

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

.. _creating_synapses:

Creating synapses
-----------------
Creating a `Synapses` instance does not create synapses, it only specifies their dynamics.
The following command creates a synapse between neuron ``5`` in the source group and
neuron ``10`` in the target group::

    S.connect(i=5, j=10)

Multiple synaptic connections can be created in a single statement::

    S.connect()
    S.connect(i=[1, 2], j=[3, 4])
    S.connect(i=numpy.arange(10), j=1)

The first statement connects all neuron pairs.
The second statement creates synapses between neurons 1 and 3, and between neurons 2 and 4.
The third statement creates synapses between the first ten neurons in the source group and neuron 1
in the target group.

It is also possible to create several synapses for a given pair of neurons::

    S.connect(i=numpy.arange(10), j=1, n=3)

This is useful for example if one wants to have multiple synapses with different delays. To
distinguish multiple variables connecting the same pair of neurons in synaptic expressions and
statements, you can create a variable storing the synapse index with the ``multisynaptic_index``
keyword::

    syn = Synapses(source_group, target_group, model='w : 1', on_pre='v += w',
                   multisynaptic_index='synapse_number')
    syn.connect(i=numpy.arange(10), j=1, n=3)
    syn.delay = '1*ms + synapse_number*2*ms'

One can also create synapses by giving (as a string) the condition for a pair
of neurons i and j to be connected by a synapse, e.g. you could
connect neurons that are not very far apart with::

    S.connect(condition='abs(i-j)<=5')


The string expressions can also refer to pre- or postsynaptic variables. This
can be useful for example for spatial connectivity: assuming that the pre- and
postsynaptic groups have parameters ``x`` and ``y``, storing their location, the
following statement connects all cells in a 250 um radius::

    S.connect(condition='sqrt((x_pre-x_post)**2 + (y_pre-y_post)**2) < 250*umeter')

Synapse creation can also be probabilistic by providing a ``p`` argument,
providing the connection probability for each pair of synapses::

    S.connect(p=0.1)

This connects all neuron pairs with a probability of 10%. Probabilities can
also be given as expressions, for example to implement a connection probability
that depends on distance::

    S.connect(condition='i != j',
              p='p_max*exp(-(x_pre-x_post)**2+(y_pre-y_post)**2) / (2*(125*umeter)**2)')

If this statement is applied to a `Synapses` object that connects a group to
itself, it prevents self-connections (``i != j``) and connects cells with a
probability that is modulated according to a 2-dimensional Gaussian of the
distance between the cells.

You can specify a mapping from i to any function f(i), e.g. the
simplest way to give a 1-to-1 connection would be::

    S.connect(j='i')

And the most general way of specifying a connection is using the
generator syntax, e.g. to connect neuron i to all neurons j with
0<=j<=i::

    S.connect(j='k for k in range(0, i+1)')

There are several parts to this syntax. The general form is::

    j='EXPR for VAR in RANGE if COND'

Here ``EXPR`` can be any integer-valued expression. VAR is the name
of the iteration variable (any name you like can be specified
here). The ``if COND`` part is optional and lets you give an
additional condition that has to be true for the synapse to be
created. Finally, ``RANGE`` can be either:

1. a Python ``range``, e.g. ``range(N)`` is the integers from
   0 to N-1, ``range(A, B)`` is the integers from A to B-1,
   ``range(low, high, step)`` is the integers from ``low`` to
   ``high-1`` with steps of size ``step``, or
2. it can be a random sample ``sample(N, p=0.1)`` gives a
   random sample of integers from 0 to N-1 with 10% probability
   of each integer appearing in the sample. This can have extra
   arguments like range, e.g. ``sample(low, high, step, p=0.1)``
   will give each integer in ``range(low, high, step)`` with
   probability 10%.

If you try to create an invalid synapse (i.e. connecting
neurons that are outside the correct range) then you will get
an error, e.g. you might like to try to do this to connect
each neuron to its neighbours::

    S.connect(j='i+(-1)**k for k in range(2)')

However this won't work at for ``i=0`` it gives ``j=-1`` which
is invalid. There is an option to just skip any synapses
that are outside the valid range::

    S.connect(j='i+(-1)**k for k in range(2)', skip_if_invalid=True)

How connection arguments are interpreted
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If conditions for connecting neurons are combined with both the ``n`` (number of
synapses to create) and the ``p`` (probability of a synapse) keywords, they are
interpreted in the following way:

    | For every pair i, j:
    |    if condition(i, j) is fulfilled:
    |        Evaluate p(i, j)
    |        If uniform random number between 0 and 1 < p(i, j):
    |            Create n(i, j) synapses for (i, j)

With the generator syntax ``j='EXPR for VAR in RANGE if COND'``, the interpretation is:

    | For every i:
    |     for every VAR in RANGE:
    |         j = EXPR
    |         if COND:
    |             Create n(i, j) synapses for (i, j)

Note that the arguments in ``RANGE`` can only depend on ``i`` and the values of
presynaptic variables. Similarly, the expression for ``j``, ``EXPR`` can depend
on ``i``, presynaptic variables, and on the iteration variable ``VAR``. The
condition ``COND`` can depend on anything (presynaptic and postsynaptic variables).

With the 1-to-1 mapping syntax ``j='EXPR'`` the interpretation is:

    | For every i:
    |     j = EXPR
    |     Create n(i, j) synapses for (i, j)


Efficiency considerations
^^^^^^^^^^^^^^^^^^^^^^^^^

If you are connecting a single pair of neurons, the direct form ``connect(i=5, j=10)``
is the most efficient. However, if you are connecting a number of neurons, it
will usually be more efficient to construct an array of ``i`` and ``j`` values
and have a single ``connect(i=i, j=j)`` call.

For large connections, you
should use one of the string based syntaxes where possible as this will
generate compiled low-level code that will be typically much faster than
equivalent Python code.

If you are expecting a majority of pairs of neurons to be connected, then using the
condition-based syntax is optimal, e.g. ``connect(condition='i!=j')``. However,
if relatively few neurons are being connected then the 1-to-1 mapping or generator syntax
will be better. For 1-to-1, ``connect(j='i')`` will always be faster than
``connect(condition='i==j')`` because the latter has to evaluate all ``N**2`` pairs
``(i, j)`` and check if the condition is true, whereas the former only has to do O(N)
operations.

One tricky problem is how to efficiently generate connectivity with a probability
``p(i, j)`` that depends on both i and j, since this requires ``N*N`` computations
even if the expected number of synapses is proportional to N. Some tricks for getting
around this are shown in :doc:`../examples/synapses.efficient_gaussian_connectivity`.

Accessing synaptic variables
----------------------------
Synaptic variables can be accessed in a similar way as `NeuronGroup` variables. They can be indexed
with two indexes, corresponding to the indexes of pre and postsynaptic neurons, or with string expressions (referring
to ``i`` and ``j`` as the pre-/post-synaptic indices, or to other state variables of the synapse or the connected neurons).
Here are a few examples::

    S.w[2, 5] = 1*nS
    S.w[1, :] = 2*nS
    S.w = 1*nS # all synapses assigned
    S.w[2, 3] = (1*nS, 2*nS)
    S.w[group1, group2] = "(1+cos(i-j))*2*nS"
    S.w[:, :] = 'rand()*nS'
    S.w['abs(x_pre-x_post) < 250*umetre'] = 1*nS

If multiple synapses exist between neurons, the calculation of the "multi-synaptic index" can be switched on during the
creation of the `Synapses` object::

    S = Synapses(input, neurons, 'w : 1', multisynaptic_index='k')
    S.connect('i==j', n=10)  # 1-to-1 connectivity with 10 synapses per pair

This index can then be used to set/get synapse-specific values::

    S.delay = '(k + 1)*ms)'  # Set delays between 1 and 10ms
    S.w['k<5'] = 0.5
    S.w['k>=5'] = 1

It also enables three-dimensional indexing, the following statement has the same effect as the last one above::

    S.w[:, :, 5:] = 1

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

    on_pre={'pre_transmission': 'ge+=w',
            'pre_plasticity': '''w=clip(w+Apost,0,inf)
                                 Apre+=dApre'''}

This creates two pathways with the given names (in fact, specifying ``on_pre=code``
is just a shorter syntax for ``on_pre={'pre': code}``) through which the delay
variables can be accessed.
The following statement, for example, sets the delay of the synapse between the first neurons
of the source and target groups in the ``pre_plasticity`` pathway::

	S.pre_plasticity.delay[0,0] = 3*ms

As mentioned above, ``pre`` pathways are generally executed before ``post``
pathways. The order of execution of several ``pre`` (or ``post``) pathways is
however arbitrary, and simply based on the alphabetical ordering of their names
(i.e. ``pre_plasticity`` will be executed before ``pre_transmission``). To
explicitly specify the order, set the ``order`` attribute of the pathway, e.g.::

    S.pre_transmission.order = -2

will make sure that the ``pre_transmission`` code is executed before the
``pre_plasticity`` code in each time step.

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

You can also record a synaptic variable for all synapses by passing ``record=True``.

The recorded traces can then be accessed in the usual way, again with the
possibility to index the `Synapses` object::

	plot(M.t / ms, M[0].w / nS)  # first synapse
	plot(M.t / ms, M[0, :].w / nS)  # all synapses originating from neuron 0
	plot(M.t / ms, M['w>0'].w / nS)  # all synapses with non-zero weights (at this time)

Note that the use of the `Synapses` object for indexing and ``record=True`` only
work in the default runtime modes. In standalone mode (see :ref:`cpp_standalone`),
the synapses have not yet been created at this point, so Brian cannot calculate
the indices.
