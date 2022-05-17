Synapses
========

.. sidebar:: For Brian 1 users

    `Synapses` is now the only class for defining synaptic interactions,
    it replaces *Connection*, *STDP*, etc. See the document
    :doc:`../introduction/brian1_to_2/synapses` for details how to convert
    Brian 1 code.

.. contents::
    :local:
    :depth: 1

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
Note that this does not create synapses (see `Creating Synapses`_), only the
synaptic models.

To define more complex models, models can be described as string equations,
similar to the models specified in `NeuronGroup`::

  S = Synapses(P, Q, model='w : volt', on_pre='v += w')

The above specifies a parameter ``w``, i.e. a synapse-specific weight. Note that
to avoid confusion, synaptic variables cannot have the same name as a pre-
or post-synaptic variables.

Synapses can also specify code that should be executed whenever a postsynaptic
spike occurs (keyword ``on_post``) and a fixed (pre-synaptic) delay for all
synapses (keyword ``delay``).

As shown above, variable names that are not referring to a synaptic variable
are automatically understood to be post-synaptic variables. To explicitly
specify that a variable should be from a pre- or post-synaptic neuron, append
the suffix ``_pre`` or ``_post``. An alternative but equivalent formulation of
the ``on_pre`` statement above would therefore be ``v_post += w``.

.. _synapse_model_syntax:

Model syntax
~~~~~~~~~~~~

The model follows exactly the same syntax as for `NeuronGroup`. There can be parameters
(e.g. synaptic variable ``w`` above), but there can also be named
subexpressions and differential equations, describing the dynamics of synaptic
variables. In all cases, synaptic variables are created, one value per synapse.

Brian also automatically defines a number of synaptic variables that can be
used in equations, ``on_pre`` and ``on_post`` statements, as well as when
:ref:`assigning to other synaptic variables <accessing_synaptic_variables>`:

``i``
    The index of the pre-synaptic source of a synapse.

``j``
    The index of the post-synaptic target of a synapse.

``N``
    The total number of synapses.

``N_incoming``
    The total number of synapses connected to the post-synaptic target of a
    synapse.

``N_outgoing``
    The total number of synapses outgoing from the pre-synaptic source of a
    synapse.

``lastupdate``
    The last time this synapse has applied an ``on_pre`` or ``on_post``
    statement. There is normally no need to refer to this variable explicitly,
    it is used to implement :ref:`event_driven_updates` (see below). It is only
    defined when event-driven equations are used.

.. _event_driven_updates:

Event-driven updates
~~~~~~~~~~~~~~~~~~~~
By default, differential equations are integrated in a clock-driven fashion, as for a
`NeuronGroup`. This is potentially very time consuming, because all synapses are updated at every
timestep and Brian will therefore emit a warning. If you are sure about integrating the equations at
every timestep (e.g. because you want to record the values continuously), then you should specify
the flag ``(clock-driven)``, which will silence the warning. To ask Brian 2 to simulate differential
equations in an event-driven fashion use the flag ``(event-driven)``. A typical example is pre- and
postsynaptic traces in STDP::

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
~~~~~~~~~~~~~~~~~~
The ``on_pre`` code is executed at each synapse receiving a presynaptic spike. For example::

	on_pre='v+=w'

adds the value of synaptic variable ``w`` to postsynaptic variable ``v``.
Any sort of code can be executed. For example, the following code defines
stochastic synapses, with a synaptic weight ``w`` and transmission probability ``p``::

	S=Synapses(neuron_input,neurons,model="""w : 1
                                      p : 1""",
        	                 on_pre="v+=w*(rand()<p)")

The code means that ``w`` is added to ``v`` with probability ``p``.
The code may also include multiple lines.

Similarly, the ``on_post`` code is executed at each synapse where the postsynaptic neuron
has fired a spike.

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

Conditional
~~~~~~~~~~~

One can also create synapses by giving (as a string) the condition for a pair
of neurons i and j to be connected by a synapse, e.g. you could
connect neurons that are not very far apart with::

    S.connect(condition='abs(i-j)<=5')


The string expressions can also refer to pre- or postsynaptic variables. This
can be useful for example for spatial connectivity: assuming that the pre- and
postsynaptic groups have user-defined parameters ``x`` and ``y``, storing their
location, the following statement connects all cells in a 250 um radius::

    S.connect(condition='sqrt((x_pre-x_post)**2 + (y_pre-y_post)**2) < 250*umeter')

Probabilistic
~~~~~~~~~~~~~

Synapse creation can also be probabilistic by providing a ``p`` argument,
providing the connection probability for each pair of synapses::

    S.connect(p=0.1)

This connects all neuron pairs with a probability of 10%. Probabilities can
also be given as expressions, for example to implement a connection probability
that depends on distance::

    S.connect(condition='i != j',
              p='p_max*exp(-(x_pre-x_post)**2+(y_pre-y_post)**2 / (2*(125*umeter)**2))')

If this statement is applied to a `Synapses` object that connects a group to
itself, it prevents self-connections (``i != j``) and connects cells with a
probability that is modulated according to a 2-dimensional Gaussian of the
distance between the cells computed from the user-defined parameters ``x``
and ``y``, storing their location.

One-to-one
~~~~~~~~~~

You can specify a mapping from i to any function f(i), e.g. the
simplest way to give a 1-to-1 connection would be::

    S.connect(j='i')

This mapping can also use a restricting condition with ``if``, e.g. to connect
neurons 0, 2, 4, 6, ... to neurons 0, 1, 2, 3, ... you could write::

    S.connect(j='int(i/2) if i % 2 == 0')

The connections above describe the target indices ``j`` as a function of the source indices ``i``.
You can also apply the syntax in the other direction, i.e. describe source indices ``i`` as a function
of target indices ``j``. For a 1-to-1 connection, this does not change anything in most cases::

    S.connect(i='j')

Note that there is a subtle difference between the two descriptions if the two groups do not have the same size:
if the source group has fewer neurons than the target group, then using `j='i'` is possible (there is a target
neuron for each source neuron), but `i='j'` would raise an error; the opposite is true if the source group is
bigger than the target group.

The second example from above (neurons 0, 2, 4, ... to neurons 0, 1, 2, ...) can be adapted for the other
direction, as well, and is possibly more intuitive in this case::

    S.connect(i='j*2')

.. _accessing_synaptic_variables:

Accessing synaptic variables
----------------------------
Synaptic variables can be accessed in a similar way as `NeuronGroup` variables. They can be indexed
with two indexes, corresponding to the indexes of pre and postsynaptic neurons, or with string expressions (referring
to ``i`` and ``j`` as the pre-/post-synaptic indices, or to other state variables of the synapse or the connected neurons).
Note that setting a synaptic variable always refers to the synapses that *currently exist*, i.e. you have to set them
*after* the relevant `Synapses.connect` call.

Here are a few examples::

    S.w[2, 5] = 1*nS
    S.w[1, :] = 2*nS
    S.w = 1*nS # all synapses assigned
    S.w[2, 3] = (1*nS, 2*nS)
    S.w[group1, group2] = "(1+cos(i-j))*2*nS"
    S.w[:, :] = 'rand()*nS'
    S.w['abs(x_pre-x_post) < 250*umetre'] = 1*nS

Assignments can also refer to :ref:`pre-defined variables <synapse_model_syntax>`,
e.g. to normalize synaptic weights.  For example, after the following assignment
the sum of weights of all synapses that a neuron receives is identical to 1,
regardless of the number of synapses it receives::

    syn.w = '1.0/N_incoming'

Note that it is also possible to index synaptic variables with a single index
(integer, slice, or array), but in this case synaptic indices have to be
provided.

The ``N_incoming`` and ``N_outgoing`` variables give access to the
total number of incoming/outgoing synapses for a neuron, but this access is given
for each *synapse*. This is necessary to apply it to individual synapses as in
the statement to normalize synaptic weights mentioned above. To access these
values per *neuron* instead, `~.Synapses.N_incoming_post` and
`~.Synapses.N_outgoing_pre` can be used. Note that synaptic equations or
``on_pre``/``on_post`` statements should always refer to ``N_incoming`` and
``N_outgoing`` without ``pre``/``post`` suffix.

Here's a little example illustrating the use of these variables::

    >>> group1 = NeuronGroup(3, '')
    >>> group2 = NeuronGroup(3, '')
    >>> syn = Synapses(group1, group2)
    >>> syn.connect(i=[0, 0, 1, 2], j=[1, 2, 2, 2])
    >>> print(syn.N_outgoing_pre)  # for each presynaptic neuron
    [2 1 1]
    >>> print(syn.N_outgoing[:])  # same numbers, but indexed by synapse
    [2 2 1 1]
    >>> print(syn.N_incoming_post)
    [0 1 3]
    >>> print(syn.N_incoming[:])
    [1 3 3 3]

Note that `~.Synapses.N_incoming_post` and `~.Synapses.N_outgoing_pre` can contain zeros for neurons
that do not have any incoming respectively outgoing synapses. In contrast, `~.Synapses.N_incoming`
and `~.Synapses.N_outgoing` will never contain zeros, because unconnected neurons are not represented
in the list of synapses.

Delays
------
There is a special synaptic variable that is automatically created: ``delay``. It is the propagation delay
from the presynaptic neuron to the synapse, i.e., the presynaptic delay. This
is just a convenience syntax for accessing the delay stored in the presynaptic
pathway: ``pre.delay``. When there is a  postsynaptic code (keyword ``post``),
the delay of the postsynaptic pathway can be accessed as ``post.delay``.

The delay variable(s) can be set and accessed in the same way as other synaptic
variables. The same semantics as for other synaptic variables apply, which means
in particular that the delay is only set for the synapses that have been already
created with `Synapses.connect`. If you want to set a global delay for all
synapses of a `Synapses` object, you can directly specify that delay as part
of the `Synapses` initializer::

    synapses = Synapses(sources, targets, '...', on_pre='...', delay=1*ms)

When you use this syntax, you can still change the delay afterwards by setting
``synapses.delay``, but you can only set it to another scalar value. If you need
different delays across synapses, do not use this syntax but instead set the
delay variable as any other synaptic variable (see above).

Monitoring synaptic variables
-----------------------------
A `StateMonitor` object can be used to monitor synaptic variables. For example, the following statement
creates a monitor for variable ``w`` for the synapses 0 and 1::

	M = StateMonitor(S, 'w', record=[0,1])

Note that these are *synapse* indices, not neuron indices. More convenient is
to directly index the `Synapses` object, Brian will automatically calculate the
indices for you in this case::

	M = StateMonitor(S, 'w', record=S[0, :])  # all synapses originating from neuron 0
	M = StateMonitor(S, 'w', record=S['i!=j'])  # all synapses excluding autapses
	M = StateMonitor(S, 'w', record=S['w>0'])  # all synapses with non-zero weights (at this time)

You can also record a synaptic variable for all synapses by passing ``record=True``.

The recorded traces can then be accessed in the usual way, again with the
possibility to index the `Synapses` object::

	plot(M.t / ms, M[S[0]].w / nS)  # first synapse
	plot(M.t / ms, M[S[0, :]].w / nS)  # all synapses originating from neuron 0
	plot(M.t / ms, M[S['w>0*nS']].w / nS)  # all synapses with non-zero weights (at this time)

Note (for users of Brian's advanced standalone mode only):
the use of the `Synapses` object for indexing and ``record=True`` only
work in the default runtime modes. In standalone mode (see :ref:`cpp_standalone`),
the synapses have not yet been created at this point, so Brian cannot calculate
the indices.

.. admonition:: The following topics are not essential for beginners.

    |

Synaptic connection/weight matrices
-----------------------------------

Brian does not directly support specifying synapses by using a
matrix, you always have to use a "sparse" format, where each
connection is defined by its source and target indices. However,
you can easily convert between the two formats. Assuming you have
a connection matrix :math:`C` of size :math:`N \times M`, where
:math:`N` is the number of presynaptic cells, and :math:`M` the
number of postsynaptic cells, with each entry being 1 for a
connection, and 0 otherwise. You can convert this matrix to arrays of
source and target indices, which you can then provide to Brian's
`~.Synapses.connect` function::

    C = ...  # The connection matrix as a numpy array of 0's and 1's
    sources, targets = C.nonzero()
    synapses = Synapses(...)
    synapses.connect(i=sources, j=targets)

Similarly, you can transform the flat array of values stored in a
synapse into a matrix form. For example, to get a matrix with all
the weight values ``w``, with ``NaN`` values where no synapse
exists::

    synapses = Synapses(source_group, target_group,
                        '''...
                           w : 1  # synaptic weight''', ...)
    # ...
    # Run e.g. a simulation with plasticity that changes the weights
    run(...)
    # Create a matrix to store the weights and fill it with NaN
    W = np.full((len(source_group), len(target_group)), np.nan)
    # Insert the values from the Synapses object
    W[synapses.i[:], synapses.j[:]] = synapses.w[:]

.. _generator_syntax:

Creating synapses with the generator syntax
-------------------------------------------

The most general way of specifying a connection is using the
generator syntax, e.g. to connect neuron i to all neurons j with
0<=j<=i::

    S.connect(j='k for k in range(0, i+1)')

There are several parts to this syntax. The general form is::

    j='EXPR for VAR in RANGE if COND'

or::

    i='EXPR for VAR in RANGE if COND'

Here ``EXPR`` can be any integer-valued expression. VAR is the name
of the iteration variable (any name you like can be specified
here). The ``if COND`` part is optional and lets you give an
additional condition that has to be true for the synapse to be
created. Finally, ``RANGE`` can be either:

1. a Python ``range``, e.g. ``range(N)`` is the integers from
   0 to N-1, ``range(A, B)`` is the integers from A to B-1,
   ``range(low, high, step)`` is the integers from ``low`` to
   ``high-1`` with steps of size ``step``;
2. a random sample ``sample(N, p=0.1)`` gives a
   random sample of integers from 0 to N-1 with 10% probability
   of each integer appearing in the sample. This can have extra
   arguments like range, e.g. ``sample(low, high, step, p=0.1)``
   will give each integer in ``range(low, high, step)`` with
   probability 10%;
3. a random sample ``sample(N, size=10)`` with a fixed size,
   in this example 10 values chosen (without replacement) from
   the integers from 0 to N-1. As for the random sample based on
   a probability, the ``sample`` expression can take additional
   arguments to sample from a restricted range.

If you try to create an invalid synapse (i.e. connecting
neurons that are outside the correct range) then you will get
an error, e.g. you might like to try to do this to connect
each neuron to its neighbours::

    S.connect(j='i+(-1)**k for k in range(2)')

However this won't work at for ``i=0`` it gives ``j=-1`` which
is invalid. There is an option to just skip any synapses
that are outside the valid range::

    S.connect(j='i+(-1)**k for k in range(2)', skip_if_invalid=True)

You can also use this argument to deal with random samples of
incorrect size, i.e. a negative size or a size bigger than the
total population size. With ``skip_if_invalid=True``, no error will
be raised and a size of 0 or the population size will be used.

Summed variables
----------------
In many cases, the postsynaptic neuron has a variable that represents a sum of variables over all
its synapses. This is called a "summed variable". An example is nonlinear synapses (e.g. NMDA)::

    neurons = NeuronGroup(1, model='''dv/dt=(gtot-v)/(10*ms) : 1
                                      gtot : 1''')
    S = Synapses(neuron_input, neurons,
                 model='''dg/dt=-a*g+b*x*(1-g) : 1
                          gtot_post = g : 1  (summed)
                          dx/dt=-c*x : 1
                          w : 1 # synaptic weight''', on_pre='x+=w')

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

Note that you cannot target the same post-synaptic variable from more than one
`Synapses` object. To work around this restriction, use multiple post-synaptic
variables that ar then summed up::

    neurons = NeuronGroup(1, model='''dv/dt=(gtot-v)/(10*ms) : 1
                                      gtot = gtot1 + gtot2: 1
                                      gtot1 : 1
                                      gtot2 : 1''')
    S1 = Synapses(neuron_input, neurons,
                  model='''dg/dt=-a1*g+b1*x*(1-g) : 1
                           gtot1_post = g : 1  (summed)
                           dx/dt=-c1*x : 1
                           w : 1 # synaptic weight
                        ''', on_pre='x+=w')
    S2 = Synapses(neuron_input, neurons,
                  model='''dg/dt=-a2*g+b2*x*(1-g) : 1
                           gtot2_post = g : 1  (summed)
                           dx/dt=-c2*x : 1
                           w : 1 # synaptic weight
                        ''', on_pre='x+=w')

Creating multi-synapses
-----------------------

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

This index can then be used to set/get synapse-specific values::

    S.delay = '(synapse_number + 1)*ms)'  # Set delays between 1 and 10ms
    S.w['synapse_number<5'] = 0.5
    S.w['synapse_number>=5'] = 1

It also enables three-dimensional indexing, the following statement has the same effect as the last one above::

    S.w[:, :, 5:] = 1

Multiple pathways
-----------------

It is possible to have multiple pathways with different update codes from the same presynaptic neuron group.
This may be interesting in cases when different operations must be applied at different times for the same
presynaptic spike, e.g. for a STDP rule that shifted in time. To do this, specify a dictionary of pathway names and codes::

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
pathways. The order of execution of several ``pre`` (or ``post``) pathways with the
same delay is however arbitrary, and simply based on the alphabetical ordering of their names
(i.e. ``pre_plasticity`` will be executed before ``pre_transmission``). To
explicitly specify the order, set the ``order`` attribute of the pathway, e.g.::

    S.pre_transmission.order = -2

will make sure that the ``pre_transmission`` code is executed before the
``pre_plasticity`` code in each time step.

Multiple pathways can also be useful for abstract models of synaptic currents, e.g.
modelling them as rectangular currents::

    synapses = Synapses(...,
                        on_pre={'up': 'I_syn_post += 1*nA',
                                'down': 'I_syn_post -= 1*nA'},
                        delay={'up': 0*ms, 'down': 5*ms}  # 5ms-wide rectangular current
                        )

Numerical integration
---------------------

Differential equation flags
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the integration of differential equations, one can use the same keywords as
for `NeuronGroup`.

.. note:: Declaring a subexpression as ``(constant over dt)`` means that it will
   be evaluated each timestep for all synapses, potentially a very costly
   operation.

Explicit event-driven updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As mentioned above, it is possible to write event-driven update code for the synaptic variables.
This can also be done manually, by defining the variable ``lastupdate`` and
referring to the predefined variable ``t`` (current time).
Here's an example for short-term plasticity::

	S=Synapses(neuron_input,neuron,
	           model='''x : 1
	                    u : 1
	                    w : 1
	                    lastupdate : second''',
	           on_pre='''u=U+(u-U)*exp(-(t-lastupdate)/tauf)
	                  x=1+(x-1)*exp(-(t-lastupdate)/taud)
	                  i+=w*u*x
	                  x*=(1-u)
	                  u+=U*(1-u)
	                  lastupdate = t''')

By default, the ``pre`` pathway is executed before the ``post`` pathway (both
are executed in the ``'synapses'`` scheduling slot, but the ``pre`` pathway has
the ``order`` attribute -1, wheras the ``post`` pathway has ``order`` 1. See
:ref:`scheduling` for more details).

Note that using the automatic ``event-driven`` approach from above is usually preferable,
see :doc:`../examples/frompapers.Stimberg_et_al_2018.example_1_COBA` for an ``event-driven``
implementation of short-term plasticity.

Technical notes
---------------

How connection arguments are interpreted
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If conditions for connecting neurons are combined with both the ``n`` (number of
synapses to create) and the ``p`` (probability of a synapse) keywords, they are
interpreted in the following way:

    | For every pair i, j:
    |    if condition(i, j) is fulfilled:
    |        Evaluate p(i, j)
    |        If uniform random number between 0 and 1 < p(i, j):
    |            Create n(i, j) synapses for (i, j)

With the generator syntax ``j='EXPR for VAR in RANGE if COND'`` (where the
``RANGE`` can be a full range or a random sample as described above), the interpretation
is:

    | For every i:
    |     for every VAR in RANGE:
    |         j = EXPR
    |         if COND:
    |             Create n(i, j) synapses for (i, j)

Note that the arguments in ``RANGE`` can only depend on ``i`` and the values of
presynaptic variables. Similarly, the expression for ``j``, ``EXPR`` can depend
on ``i``, presynaptic variables, and on the iteration variable ``VAR``. The
condition ``COND`` can depend on anything (presynaptic and postsynaptic variables).

The generator syntax expressing ``i`` as a function of ``j`` is interpreted
in the same way:

    | For every j:
    |     for every VAR in RANGE:
    |         i = EXPR
    |         if COND:
    |             Create n(i, j) synapses for (i, j)

Here, ``RANGE`` can only depend on ``j`` and postsynaptic variables, and ``EXPR``
can only depend on ``j``, postsynaptic variables, and on the iteration variable
``VAR``.

With the 1-to-1 mapping syntax ``j='EXPR'`` the interpretation is:

    | For every i:
    |     j = EXPR
    |     Create n(i, j) synapses for (i, j)

And finally, ``i='EXPR'`` is interpreted as:

    | For every j:
    |     i = EXPR
    |     Create n(i, j) synapses for (i, j)

Efficiency considerations
~~~~~~~~~~~~~~~~~~~~~~~~~

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
