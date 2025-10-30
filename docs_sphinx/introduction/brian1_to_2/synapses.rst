Synapses (Brian 1 --> 2 conversion)
===================================
.. sidebar:: Brian 2 documentation

    For the main documentation about defining and creating synapses, see the
    document :doc:`../../user/synapses`.

.. contents::
    :local:
    :depth: 1

Converting Brian 1's ``Connection`` class
-----------------------------------------
In Brian 2, the `Synapses` class is the only class to model synaptic
connections, you will therefore have to convert all uses of Brian 1's
``Connection`` class. The ``Connection`` class increases a post-synaptic
variable by a certain amount (the "synaptic weight") each time a pre-synaptic
spike arrives. This has to be explicitly specified when using the `Synapses`
class, the equivalent to the basic ``Connection`` usage is:

+----------------------------------------------+---------------------------------------------------+
| Brian 1                                      | Brian 2                                           |
+==============================================+===================================================+
+ .. code::                                    | .. code::                                         |
+                                              |                                                   |
+    conn = Connection(source, target, 'ge')   |    conn = Synapses(source, target, 'w : siemens', |
+                                              |                    on_pre='ge += w')              |
+                                              |                                                   |
+----------------------------------------------+---------------------------------------------------+

Note that he variable ``w``, which stores the synaptic weight, has to have the
same units as the post-synaptic variable (in this case: ``ge``) that it
increases.

Creating synapses and setting weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the ``Connection`` class, creating a synapse and setting its weight is a
single process whereas with the `Synapses` class those two steps are separate.
There is no direct equivalent to the convenience functions ``connect_full``,
``connect_random`` and ``connect_one_to_one``, but you can easily implement
the same functionality with the general mechanism of `Synapses.connect`:

+----------------------------------------------+---------------------------------------------------+
| Brian 1                                      | Brian 2                                           |
+==============================================+===================================================+
+ .. code::                                    | .. code::                                         |
+                                              |                                                   |
+    conn1 = Connection(source, target, 'ge')  |    conn1 = Synapses(source, target, 'w: siemens', |
+    conn1[3, 5] = 3*nS                        |                     on_pre='ge += w')             |
+                                              |    conn1.connect(i=3, j=5)                        |
+                                              |    conn1.w[3, 5] = 3*nS  # (or conn1.w = 3*nS)    |
+                                              |                                                   |
+----------------------------------------------+---------------------------------------------------+
+ .. code::                                    | .. code::                                         |
+                                              |                                                   |
+    conn2 = Connection(source, target, 'ge')  |    conn2 = ... # see above                        |
+    conn2.connect_full(source, target, 5*nS)  |    conn2.connect()                                |
+                                              |    conn2.w = 5*nS                                 |
+                                              |                                                   |
+----------------------------------------------+---------------------------------------------------+
+ .. code::                                    | .. code::                                         |
+                                              |                                                   |
+    conn3 = Connection(source, target, 'ge')  |    conn3 = ... # see above                        |
+    conn3.connect_random(source, target,      |    conn3.connect(p=0.02)                          |
+                         sparseness=0.02,     |    conn3.w = 2*nS                                 |
+                         weight=2*ns)         |                                                   |
+                                              |                                                   |
+----------------------------------------------+---------------------------------------------------+
+ .. code::                                    | .. code::                                         |
+                                              |                                                   |
+    conn4 = Connection(source, target, 'ge')  |    conn4 = ... # see above                        |
+    conn4.connect_one_to_one(source, target,  |    conn4.connect(j='i')                           |
+                             weight=4*nS)     |    conn4.w = 4*nS                                 |
+                                              |                                                   |
+----------------------------------------------+---------------------------------------------------+
+ .. code::                                    | .. code::                                         |
+                                              |                                                   |
+    conn5 = IdentityConnection(source, target,|    conn5 = Synapses(source, target,               |
+                               weight=3*nS)   |                     'w : siemens (shared)')       |
+                                              |    conn5.w = 3*nS                                 |
+                                              |                                                   |
+----------------------------------------------+---------------------------------------------------+

Weight matrices
~~~~~~~~~~~~~~~

Brian 2's `Synapses` class does not support setting the weights of a neuron with
a weight matrix. However, `Synapses.connect` creates the synapses in a
predictable order (first all synapses for the first pre-synaptic cell, then all
synapses for the second pre-synaptic cell, etc.), so a reshaped "flat" weight
matrix can be used:

+----------------------------------------------+---------------------------------------------------+
| Brian 1                                      | Brian 2                                           |
+==============================================+===================================================+
+ .. code::                                    | .. code::                                         |
+                                              |                                                   |
+    # len(source) == 20, len(target) == 30    |    # len(source) == 20, len(target) == 30         |
+    conn6 = Connection(source, target, 'ge')  |    conn6 = Synapses(source, target, 'w: siemens', |
+    W = rand(20, 30)*nS                       |                     on_pre='ge += w')             |
+    conn6.connect(source, target, weight=W)   |    W = rand(20, 30)*nS                            |
+                                              |    conn6.connect()                                |
+                                              |    conn6.w = W.flatten()                          |
+                                              |                                                   |
+----------------------------------------------+---------------------------------------------------+

However note that if your weight matrix can be described mathematically (e.g.
random as in the example above), then you should not create a weight matrix in
the first place but use Brian 2's mechanism to set variables based on
mathematical expressions (in the above case: ``conn5.w = 'rand()'``). Especially
for big connection matrices this will have better performance, since it will be
executed in generated code. You should only resort to explicit weight matrices
when there is no alternative (e.g. to load weights from previous simulations).

In Brian 1, you can restrict the functions ``connect``, ``connect_random``, etc.
to subgroups. Again, there is no direct equivalent to this in Brian 2, but the
general string syntax allows you to make connections conditional on logical
statements that refer to pre-/post-synaptic indices and can therefore also used
to restrict the connection to a subgroup of cells. When you set the synaptic
weights, you *can* however use subgroups to restrict the subset of weights you
want to set.

+--------------------------------------------------------+---------------------------------------------------+
| Brian 1                                                | Brian 2                                           |
+========================================================+===================================================+
+ .. code::                                              | .. code::                                         |
+                                                        |                                                   |
+    conn7 = Connection(source, target, 'ge')            |    conn7 = Synapses(source, target, 'w: siemens', |
+    conn7.connect_full(source[:5], target[5:10], 5*nS)  |                     on_pre='ge += w')             |
+                                                        |    conn7.connect('i < 5 and j >=5 and j <10')     |
+                                                        |    # Alternative (more efficient):                |
+                                                        |    # conn7.connect(j='k in range(5, 10) if i < 5')|
+                                                        |    conn7.w[source[:5], target[5:10]] = 5*nS       |
+                                                        |                                                   |
+--------------------------------------------------------+---------------------------------------------------+

Connections defined by functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Brian 1 allowed you to pass in a function as the value for the weight
argument in a ``connect`` call (and also for the sparseness argument in
``connect_random``). You should be able to replace such use cases by the the
general, string-expression based method:

+------------------------------------------------------------------+---------------------------------------------------+
| Brian 1                                                          | Brian 2                                           |
+==================================================================+===================================================+
+ .. code::                                                        | .. code::                                         |
+                                                                  |                                                   |
+    conn8 = Connection(source, target, 'ge')                      |    conn8 = Synapses(source, target, 'w: siemens', |
+    conn8.connect_full(source, target,                            |                     on_pre='ge += w')             |
+                       weight=lambda i,j:(1+cos(i-j))*2*nS)       |    conn8.connect()                                |
+                                                                  |    conn8.w = '(1 + cos(i - j))*2*nS'              |
+                                                                  |                                                   |
+------------------------------------------------------------------+---------------------------------------------------+
+ .. code::                                                        | .. code::                                         |
+                                                                  |                                                   |
+    conn9 = Connection(source, target, 'ge')                      |    conn9 = ... # see above                        |
+    conn9.connect_random(source, target,                          |    conn9.connect(p=0.02)                          |
+                         sparseness=0.02,                         |    conn9.w = 'rand()*nS'                          |
+                         weight=lambda:rand()*nS)                 |                                                   |
+                                                                  |                                                   |
+------------------------------------------------------------------+---------------------------------------------------+
+ .. code::                                                        | .. code::                                         |
+                                                                  |                                                   |
+    conn10 = Connection(source, target, 'ge')                     |    conn10 = ... # see above                       |
+    conn10.connect_random(source, target,                         |    conn10.connect(p='exp(-abs(i - j)*.1)')        |
+                          sparseness=lambda i,j:exp(-abs(i-j)*.1),|    conn10.w = 2*nS                                |
+                          weight=2*ns)                            |                                                   |
+                                                                  |                                                   |
+------------------------------------------------------------------+---------------------------------------------------+

Delays
~~~~~~
The specification of delays changed in several aspects from Brian 1 to Brian 2:
In Brian 1, delays where homogeneous by default, and heterogeneous delays had
to be marked by ``delay=True``, together with the specification of the maximum
delay. In Brian 2, heterogeneous delays are the default and you do not have to
state the maximum delay. Brian 1's syntax of specifying a pair of values to get
randomly distributed delays in that range is no longer supported, instead use
Brian 2's standard string syntax:

+----------------------------------------------------------+-----------------------------------------------------+
| Brian 1                                                  | Brian 2                                             |
+==========================================================+=====================================================+
+ .. code::                                                | .. code::                                           |
+                                                          |                                                     |
+    conn11 = Connection(source, target, 'ge', delay=True, |    conn11 = Synapses(source, target, 'w : siemens', |
+                        max_delay=5*ms)                   |                      on_pre='ge += w')              |
+    conn11.connect_full(source, target, weight=3*nS,      |    conn11.connect()                                 |
+                        delay=(0*ms, 5*ms))               |    conn11.w = 3*nS                                  |
+                                                          |    conn11.delay = 'rand()*5*ms'                     |
+                                                          |                                                     |
+----------------------------------------------------------+-----------------------------------------------------+

Modulation
~~~~~~~~~~
In Brian 2, there's no need for the ``modulation`` keyword that Brian 1 offered,
you can describe the modulation as part of the ``on_pre`` action:

+----------------------------------------------------------+-----------------------------------------------------+
| Brian 1                                                  | Brian 2                                             |
+==========================================================+=====================================================+
+ .. code::                                                | .. code::                                           |
+                                                          |                                                     |
+    conn12 = Connection(source, target, 'ge',             |    conn12 = Synapses(source, target, 'w : siemens', |
+                        modulation='u')                   |                      on_pre='ge += w * u_pre')      |
+                                                          |                                                     |
+----------------------------------------------------------+-----------------------------------------------------+

Structure
~~~~~~~~~
There's no equivalen for Brian 1's ``structure`` keyword in Brian 2, synapses
are always stored in a sparse data structure. There is currently no support for
changing synapses at run time (i.e. the "dynamic" structure of Brian 1).


Converting Brian 1's ``Synapses`` class
---------------------------------------
Brian 2's `Synapses` class works for the most part like the class of the same
name in Brian 1. There are however some differences in details, listed below:

Synaptic models
~~~~~~~~~~~~~~~
The basic syntax to define a synaptic model is unchanged, but the keywords
``pre`` and ``post`` have been renamed to ``on_pre`` and ``on_post``,
respectively.

+----------------------------------------------------------------------------+----------------------------------------------------------------------------+
| Brian 1                                                                    | Brian 2                                                                    |
+============================================================================+============================================================================+
| .. code::                                                                  | .. code::                                                                  |
|                                                                            |                                                                            |
|    stdp_syn = Synapses(inputs, neurons, model='''                          |    stdp_syn = Synapses(inputs, neurons, model='''                          |
|                        w:1                                                 |                        w:1                                                 |
|                        dApre/dt = -Apre/taupre : 1 (event-driven)          |                        dApre/dt = -Apre/taupre : 1 (event-driven)          |
|                        dApost/dt = -Apost/taupost : 1 (event-driven)''',   |                        dApost/dt = -Apost/taupost : 1 (event-driven)''',   |
|                        pre='''ge + =w                                      |                        on_pre='''ge + =w                                   |
|                               Apre += delta_Apre                           |                               Apre += delta_Apre                           |
|                               w = clip(w + Apost, 0, gmax)''',             |                               w = clip(w + Apost, 0, gmax)''',             |
|                        post='''Apost += delta_Apost                        |                        on_post='''Apost += delta_Apost                     |
|                                w = clip(w + Apre, 0, gmax)''')             |                                w = clip(w + Apre, 0, gmax)''')             |
|                                                                            |                                                                            |
+----------------------------------------------------------------------------+----------------------------------------------------------------------------+

Lumped variables (summed variables)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The syntax to define lumped variables (we use the term "summed variables" in
Brian 2) has been changed: instead of assigning the synaptic variable to the
neuronal variable you'll have to include the summed variable in the synaptic
equations with the flag ``(summed)``:

+------------------------------------------------------------+------------------------------------------------------------+
| Brian 1                                                    | Brian 2                                                    |
+============================================================+============================================================+
| .. code::                                                  | .. code::                                                  |
|                                                            |                                                            |
|     # a non-linear synapse (e.g. NMDA)                     |     # a non-linear synapse (e.g. NMDA)                     |
|     neurons = NeuronGroup(1, model='''                     |     neurons = NeuronGroup(1, model='''                     |
|                           dv/dt = (gtot - v)/(10*ms) : 1   |                           dv/dt = (gtot - v)/(10*ms) : 1   |
|                           gtot : 1''')                     |                           gtot : 1''')                     |
|     syn = Synapses(inputs, neurons,                        |     syn = Synapses(inputs, neurons,                        |
|                    model='''                               |                    model='''                               |
|                    dg/dt = -a*g+b*x*(1-g) : 1              |                    dg/dt = -a*g+b*x*(1-g) : 1              |
|                    dx/dt = -c*x : 1                        |                    dx/dt = -c*x : 1                        |
|                    w : 1 # synaptic weight''',             |                    w : 1 # synaptic weight                 |
|                    pre='x += w')                           |                    gtot_post = g : 1 (summed)''',          |
|     neurons.gtot=S.g                                       |                    on_pre='x += w')                        |
|                                                            |                                                            |
+------------------------------------------------------------+------------------------------------------------------------+

Creating synapses
~~~~~~~~~~~~~~~~~
In Brian 1, synapses were created by assigning ``True`` or an integer (the
number of synapses) to an indexed `Synapses` object. In Brian 2, all synapse
creation goes through the `Synapses.connect` function. For examples how to
create more complex connection patterns, see the section on translating
``Connections`` objects above.

+-------------------------------+-------------------------------+
| Brian 1                       | Brian 2                       |
+===============================+===============================+
| .. code::                     | .. code::                     |
|                               |                               |
|    syn = Synapses(...)        |    syn = Synapses(...)        |
|    # single synapse           |    # single synapse           |
|    syn[3, 5] = True           |    syn.connect(i=3, j=5)      |
|                               |                               |
+-------------------------------+-------------------------------+
| .. code::                     | .. code::                     |
|                               |                               |
|    # all-to-all connections   |    # all-to-all connections   |
|    syn[:, :] = True           |    syn.connect()              |
|                               |                               |
+-------------------------------+-------------------------------+
| .. code::                     | .. code::                     |
|                               |                               |
|    # all to neuron number 1   |    # all to neuron number 1   |
|    syn[:, 1] = True           |    syn.connect(j='1')         |
|                               |                               |
+-------------------------------+-------------------------------+
| .. code::                     | .. code::                     |
|                               |                               |
|    # multiple synapses        |    # multiple synapses        |
|    syn[4, 7] = 3              |    syn.connect(i=4, j=7, n=3) |
|                               |                               |
+-------------------------------+-------------------------------+
| .. code::                     | .. code::                     |
|                               |                               |
|    # connection probability 2%|    # connection probability 2%|
|    syn[:, :] = 0.02           |    syn.connect(p=0.02)        |
|                               |                               |
+-------------------------------+-------------------------------+

Multiple pathways
~~~~~~~~~~~~~~~~~
As Brian 1, Brian 2 supports multiple pre- or post-synaptic pathways, with
separate pre-/post-codes and delays. In Brian 1, you have to specify the
pathways as tuples and can then later access them individually by using their
index. In Brian 2, you specify the pathways as a dictionary, i.e. by giving
them individual names which you can then later use to access them (the default
pathways are called ``pre`` and ``post``):

+----------------------------------------------------------+----------------------------------------------------------+
| Brian 1                                                  | Brian 2                                                  |
+==========================================================+==========================================================+
|    .. code::                                             |    .. code::                                             |
|                                                          |                                                          |
|       S = Synapses(...,                                  |       S = Synapses(...,                                  |
|                    pre=('ge + =w',                       |                    pre={'pre_transmission':              |
|                         '''w = clip(w + Apost, 0, inf)   |                         'ge += w',                       |
|                            Apre += delta_Apre'''),       |                         'pre_plasticity':                |
|                    post='''Apost += delta_Apost          |                         '''w = clip(w + Apost, 0, inf)   |
|                            w = clip(w + Apre, 0, inf)''')|                            Apre += delta_Apre'''},       |
|                                                          |                    post='''Apost += delta_Apost          |
|       S[:, :] = True                                     |                            w = clip(w + Apre, 0, inf)''')|
|       S.delay[1][:, :] = 3*ms # delayed trace            |                                                          |
|                                                          |       S.connect()                                        |
|                                                          |       S.pre_plasticity.delay[:, :] = 3*ms # delayed trace|
|                                                          |                                                          |
+----------------------------------------------------------+----------------------------------------------------------+

Monitoring synaptic variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Both in Brian 1 and Brian 2, you can record the values of synaptic variables
with a `StateMonitor`. You no longer have to call an explicit indexing function,
but you can directly provide an appropriately indexed `Synapses` object. You
can now also use the same technique to index the `StateMonitor` object to get
the recorded values, see the respective section in the
:doc:`../../user/synapses` documentation for details.

+-------------------------------------------------+----------------------------------------------+
| Brian 1                                         | Brian 2                                      |
+=================================================+==============================================+
| .. code::                                       | .. code::                                    |
|                                                 |                                              |
|    syn = Synapses(...)                          |    syn = Synapses(...)                       |
|    # record all synapse targetting neuron 3     |    # record all synapse targetting neuron 3  |
|    indices = syn.synapse_index((slice(None), 3))|    mon = StateMonitor(S, 'w', record=S[:, 3])|
|    mon = StateMonitor(S, 'w', record=indices)   |                                              |
|                                                 |                                              |
+-------------------------------------------------+----------------------------------------------+
