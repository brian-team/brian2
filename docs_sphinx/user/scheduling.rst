Scheduling
==========

During a simulation run, many different objects responsible for the numerical
integration, the threshold and reset, the synaptic propagation, etc. are
executed. Determining which computation is performed when is called
"scheduling". The coarse scheduling deals with multiple clocks (e.g. one for
the simulation and another one with a larger timestep to records snapshots of
the activity) and follows the following pattern:

1. Determine which set of clocks to update. This will be the clock with the
   smallest value of `Clock.t`. If there are several with the same value,
   then all objects with these clocks will be updated simultaneously.
2. If the `Clock.t` value of these clocks is past the end time of the
   simulation, stop running.
3. For each object whose `BrianObject.clock` is set to one of the clocks from the
   previous steps, call the `BrianObject.update` method.
   The order in which the objects are updated is described below.
4. Increase `Clock.t` by `Clock.dt` for each of the clocks and return to
   step 1.

The fine scheduling deals with the order of objects in step 3 above. This
scheduling is responsible that even though state update (numerical integration),
thresholding and reset for a `NeuronGroup` are performed with the same `Clock`,
the state update is always performed first, followed by the thresholding and the
reset. This schedule is determined by `Network.schedule` which is a list of
strings, determining "execution slots" and their order. It defaults to:
``['start', 'groups', 'thresholds', 'synapses', 'resets', 'end']``

In which slot an object is updated is determined by its `BrianObject.when`
attribute which is set to sensible values for most objects (resets will happen
in the ``reset`` slot, etc.) but sometimes make sense to change, e.g. if one
would like a `StateMonitor`, which by default records in the ``end`` slot, to
record the membrane potential before a reset is applied (otherwise no threshold
crossings will be observed in the membrane potential traces). If two objects
fall in the same execution slot, they will be updated in ascending order
according to their `BrianObject.order` attribute, an integer number defaulting
to 0.