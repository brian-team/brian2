Networks and clocks (Brian 1 --> 2 conversion)
==============================================
.. sidebar:: Brian 2 documentation

    For the main documentation about running simulations, controling the
    simulation timestep, etc., see the document :doc:`../../user/running`.

.. contents::
    :local:
    :depth: 1

Clocks and timesteps
--------------------
Brian's system of handling clocks has substantially changed. For details about
the new system in place see :ref:`time_steps`. The main differences to Brian 1
are:

* There is no more "clock guessing" -- objects either use the `defaultclock` or
  a ``dt``/``clock`` value that was explicitly specified during their
  construction.
* In Brian 2, the time step is allowed to change after the creation of an object
  and between runs -- the relevant value is the value in place at the point of
  the `run` call.
* It is rarely necessary to create an explicit `Clock` object, most of the time
  you should use the `defaultclock` or provide a ``dt`` argument during the
  construction of the object.
* There's only one `Clock` class, the (deprecated) ``FloatClock``,
  ``RegularClock``, etc. classes that Brian 1 provided no longer exist.
* It is no longer possible to (re-)set the time of a clock explicitly, there is
  no direct equivalent of ``Clock.reinit`` and ``reinit_default_clock``. To
  start a completely new simulation after you have finished a previous one,
  either create a new `Network` or use the `start_scope` mechanism. To "rewind"
  a simulation to a previous point, use the new `store`/`restore` mechanism. For
  more details, see below and :doc:`../../user/running`.

Networks
--------
Both Brian 1 and Brian 2 offer two ways to run a simulation: either by
explicitly creating a `Network` object, or by using a `MagicNetwork`, i.e. a
simple `run` statement.

Explicit network
~~~~~~~~~~~~~~~~
The mechanism to create explicit `Network` objects has not changed significantly
from Brian 1 to Brian 2. However, creating a new `Network` will now also
automatically reset the clock back to 0s, and stricter checks no longer allow
the inclusion of the same object in multiple networks.

+------------------------------+------------------------------+
+ Brian 1                      | Brian 2                      |
+==============================+==============================+
| .. code::                    | .. code::                    |
|                              |                              |
|    group = ...               |    group = ...               |
|    mon = ...                 |    mon = ...                 |
|    net = Network(group, mon) |    net = Network(group, mon) |
|    net.run(1*ms)             |    net.run(1*ms)             |
|                              |                              |
|    reinit()                  |    # new network starts at 0s|
|    group = ...               |    group = ...               |
|    mon = ...                 |    mon = ...                 |
|    net = Network(group, mon) |    net = Network(group, mon) |
|    net.run(1*ms)             |    net.run(1*ms)             |
|                              |                              |
+------------------------------+------------------------------+

"Magic" network
~~~~~~~~~~~~~~~
For most simple, "flat", scripts (see e.g. the :doc:`../../examples/index`),
the `run` statement in Brian 2 automatically collects all the Brian objects
(`NeuronGroup`, etc.) into a "magic" network in the same way as Brian 1 did.
The logic behind this collection has changed, though, with important
consequences for more complex simulation scripts: in Brian 1, the magic network
includes all Brian objects that have been *created* in the same execution frame
as the `run` call. Objects that are created in other functions could be added
using ``magic_return`` and ``magic_register``. In Brian 2, the magic network
contains all Brian objects that are *visible* in the same execution frame as the
`run` call. The advantage of the new system is that it is clearer what will be
included in the network and there is no danger of including previously created,
but no longer needed, objects in a simulation. E.g. in the following example,
a common mistake in Brian 1 was to not include the `clear()`, which meant that
each run not only simulated the current objects, but also all objects from
previous loop iterations. Also, without the ``reinit_default_clock()``,
each run would start at the end time of the previous run. In Brian 2, this loop
does not need any explicit clearing up, each `run` will only simulate the
object that it "sees" (``group1``, ``group2``, ``syn``, and ``mon``) and start
each simulation at 0s:

+--------------------------------------------+--------------------------------------------+
| Brian 1                                    | Brian 2                                    |
+============================================+============================================+
| .. code::                                  | .. code::                                  |
|                                            |                                            |
|     for r in range(100):                   |     for r in range(100):                   |
|         reinit_default_clock()             |                                            |
|         clear()                            |                                            |
|         group1 = NeuronGroup(...)          |         group1 = NeuronGroup(...)          |
|         group2 = NeuronGroup(...)          |         group2 = NeuronGroup(...)          |
|         syn = Synapses(group1, group2, ...)|         syn = Synapses(group1, group2, ...)|
|         mon = SpikeMonitor(group2)         |         mon = SpikeMonitor(group2)         |
|         run(1*second)                      |         run(1*second)                      |
|                                            |                                            |
+--------------------------------------------+--------------------------------------------+

There is no replacement for the ``magic_return`` and ``magic_register``
functions. If the returned object is stored in a variable at the level of
the `run` call, then it is no longer necessary to use ``magic_return``, as the
returned object is "visible" at the level of the `run` call:

+-----------------------------------------------+-------------------------------------------------+
| Brian 1                                       | Brian 2                                         |
+===============================================+=================================================+
| .. code::                                     | .. code::                                       |
|                                               |                                                 |
|     @magic_return                             |                                                 |
|     def f():                                  |     def f():                                    |
|         return PoissonGroup(100, rates=100*Hz)|         return PoissonGroup(100, rates=100*Hz)  |
|                                               |                                                 |
|     pg = f() # needs magic_return             |     pg = f() # is "visible" and will be included|
|     mon = SpikeMonitor(pg)                    |     mon = SpikeMonitor(pg)                      |
|     run(100*ms)                               |     run(100*ms)                                 |
|                                               |                                                 |
+-----------------------------------------------+-------------------------------------------------+

The general recommendation is however: if your script is complex (multiple
functions/files/classes) and you are not sure whether some objects will be
included in the magic network, use an explicit `Network` object.

Note that one consequence of the "is visible" approach is that objects stored
in containers (lists, dictionaries, ...) will not be automatically included in
Brian 2. Use an explicit `Network` object to get around this restriction:

+----------------------------------------+----------------------------------------+
| Brian 1                                | Brian 2                                |
+========================================+========================================+
| .. code::                              | .. code::                              |
|                                        |                                        |
|     groups = {'exc': NeuronGroup(...), |     groups = {'exc': NeuronGroup(...), |
|               'inh': NeuronGroup(...)} |               'inh': NeuronGroup(...)} |
|     ...                                |     ...                                |
|                                        |     net = Network(groups)              |
|     run(5*ms)                          |     net.run(5*ms)                      |
|                                        |                                        |
+----------------------------------------+----------------------------------------+

External constants
~~~~~~~~~~~~~~~~~~
In Brian 2, external constants are taken from the surrounding namespace at
the point of the `run` call and not when the object is defined (for other ways
to define the namespace, see :ref:`external-variables`). This allows to easily
change external constants between runs, in contrast to Brian 1 where the whether
this worked or not depended on details of the model (e.g. whether linear
integration was used):

+----------------------------------------------------------+-----------------------------------------------------------+
| Brian 1                                                  | Brian 2                                                   |
+==========================================================+===========================================================+
| .. code::                                                | .. code::                                                 |
|                                                          |                                                           |
|    tau = 10*ms                                           |     tau = 10*ms                                           |
|    # to be sure that changes between runs are taken into |                                                           |
|    # account, define "I" as a neuronal parameter         |     # The value for I will be updated at each run         |
|    group = NeuronGroup(10, '''dv/dt = (-v + I) / tau : 1 |     group = NeuronGroup(10, 'dv/dt = (-v + I) / tau : 1') |
|                               I : 1''')                  |                                                           |
|    group.v = linspace(0, 1, 10)                          |     group.v = linspace(0, 1, 10)                          |
|    group.I = 0.0                                         |     I = 0.0                                               |
|    mon = StateMonitor(group, 'v', record=True)           |     mon = StateMonitor(group, 'v', record=True)           |
|    run(5*ms)                                             |     run(5*ms)                                             |
|    group.I = 0.5                                         |     I = 0.5                                               |
|    run(5*ms)                                             |     run(5*ms)                                             |
|    group.I = 0.0                                         |     I = 0.0                                               |
|    run(5*ms)                                             |     run(5*ms)                                             |
|                                                          |                                                           |
+----------------------------------------------------------+-----------------------------------------------------------+
