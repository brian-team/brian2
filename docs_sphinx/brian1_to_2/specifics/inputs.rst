Inputs
======
You can find the documentation for how to add external stimulation to a network
in the document :doc:`../../user/input`. This section describes the major
differences to Brian 1.

Poisson Input
-------------
Brian 2 provides the same two groups that Brian 1 provided: `PoissonGroup` and
`PoissonInput`. The mechanism for inhomogoneous Poisson processes has changed:
instead of providing a Python function of time, you'll now have to provide a
string expression that is evaluated at every time step. For most use cases, this
should allow a direct translation:

+-------------------------------------------------+------------------------------------------+
| Brian 1                                         | Brian 2                                  |
+=================================================+==========================================+
+ .. code::                                       | .. code::                                |
+                                                 |                                          |
+   rates = lambda t:(1+cos(2*pi*t*1*Hz))*10*Hz   |   rates = '(1 + cos(2*pi*t*1*Hz)*10*Hz)' |
+   group = PoissonGroup(100, rates=rates)        |   group = PoissonGroup(100, rates=rates) |
+                                                 |                                          |
+-------------------------------------------------+------------------------------------------+

For more complex rate modulations, the expression can refer to
:ref:`user_functions` and/or you can replace the `PoissonGroup` by a general
`NeuronGroup` with a threshold condition ``rand()<rates*dt`` (which allows you
to store per-neuron attributes).

There is currently no direct replacement for the more advanced features of
`PoissonInput` (``record``, ``freeze``, ``copies``, ``jitter``, and
``reliability`` keywords), but various workarounds are possible, e.g. by
directly using a `BinomialFunction` in the equations. For example, you can get
the functionality of the ``freeze`` keyword (identical Poisson events for all
neurons) by storing the input in a shared variable and then distribute the input
to all neurons:

+---------------------------------------------------+-------------------------------------------------------------+
| Brian 1                                           | Brian 2                                                     |
+===================================================+=============================================================+
+ .. code::                                         | .. code::                                                   |
+                                                   |                                                             |
+   group = NeuronGroup(10,                         |   group = NeuronGroup(10, '''dv/dt = -v / (10*ms) : 1       |
+                       'dv/dt = -v/(10*ms) : 1')   |                              shared_input : 1 (shared)''')  |
+   input = PoissonInput(group, N=1000, rate=1*Hz,  |   poisson_input = BinomialFunction(n=1000, p=1*Hz*group.dt) |
+                        weight=0.1, state='v',     |   group.run_regularly('''shared_input = poisson_input()*0.1 |
+                        freeze=True)               |                          v += shared_input''')              |
+                                                   |                                                             |
+---------------------------------------------------+-------------------------------------------------------------+

Spike generation
----------------
`SpikeGeneratorGroup` provides mostly the same functionality as in Brian 1. In
contrast to Brian 1, there is only one way to specify which neurons spike and
when -- you have to provide the index array and the times array as separate
arguments:

+----------------------------------------------------------+----------------------------------------------------+
| Brian 1                                                  | Brian 2                                            |
+==========================================================+====================================================+
| .. code::                                                | .. code::                                          |
|                                                          |                                                    |
|   gen1 = SpikeGeneratorGroup(2, [(0, 0*ms), (1, 1*ms)])  |   gen1 = SpikeGeneratorGroup(2, [0, 1], [0, 1]*ms) |
|   gen2 = SpikeGeneratorGroup(2, [(array([0, 1]), 0*ms),  |   gen2 = SpikeGeneratorGroup(2, [0, 1, 0, 1],      |
|                                  (array([0, 1]), 1*ms)]  |                              [0, 0, 1, 1]*ms)      |
|   gen3 = SpikeGeneratorGroup(2, (array([0, 1]),          |   gen3 = SpikeGeneratorGroup(2, [0, 1], [0, 1]*ms) |
|                                  array([0, 1])*ms))      |                                                    |
|   gen4 = SpikeGeneratorGroup(2, array([[0, 0.0],         |   gen4 = SpikeGeneratorGroup(2, [0, 1], [0, 1]*ms) |
|                                       [1, 0.001]])       |                                                    |
+----------------------------------------------------------+----------------------------------------------------+

.. note::

    For large arrays, make sure to provide a `Quantity` array (e.g.
    ``[0, 1, 2]*ms``) and not a list of `Quantity` values (e.g.
    ``[0*ms, 1*ms, 2*ms]``). A list has first to be translated into an array
    which can take a considerable amount of time for a list with many elements.

There is no direct equivalent of the Brian 1 option to use a generator that
updates spike times online. The easiest alternative in Brian 2 is to
pre-calculate the spikes and then use a standard `SpikeGeneratorGroup`. If this
is not possible (e.g. there are two many spikes to fit in memory), then you can
workaround the restriction by using custom code (see :ref:`user_functions` and
:ref:`network_operation`).
