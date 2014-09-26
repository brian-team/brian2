Scheduling
==========

Every simulated object in Brian has three attributes that can be specified at
object creation time: ``dt``, ``when``, and ``order``. The time step of the
simulation is determined by ``dt``, if it is specified, a new `Clock` with the
given ``dt`` will be created for the object. Alternatively, a ``clock`` object
can be specified directly, this can be useful if a clock should be shared
between several objects -- under most circumstances, however, a user should not
have to deal with the creation of `Clock` objects and just define ``dt``. If
neither a ``dt`` nor a ``clock`` argument is specified, the object will use the
`defaultclock`. Setting ``defaultclock.dt`` will therefore change the ``dt`` of
all objects that use the `defaultclock`.

Note that directly changing the ``dt`` attribute of an object is not allowed,
neither it is possible to assign to ``dt`` in abstract code statements. To
change ``dt`` between runs, change the ``dt`` attribute of the respective
`Clock` object (which is also accessible as the ``clock`` attribute of each
`BrianObject`). The ``when`` and the ``order`` attributes can be changed by
setting the respective attributes of a `BrianObject`.

During a single time step, objects are updated according to their ``when``
argument's position in the schedule.  This schedule is determined by
`Network.schedule` which is a list of strings, determining "execution slots" and
their order. It defaults to: ``['start', 'groups', 'thresholds', 'synapses',
'resets', 'end']``. The default
for the ``when`` attribute is a sensible value for most objects (resets will
happen in the ``reset`` slot, etc.) but sometimes it make sense to change it,
e.g. if one would like a `StateMonitor`, which by default records in the
``end`` slot, to record the membrane potential before a reset is applied
(otherwise no threshold crossings will be observed in the membrane potential
traces). Note that you can also add new slots to the schedule and refer to them
in the ``when`` argument of an object.

Finally, if during a time step two objects fall in the same execution
slot, they will be updated in ascending order according to their
``order`` attribute, an integer number defaulting to 0. If two objects have
the same ``when`` and ``order`` attribute then they will be updated in an
arbitrary but reproducible order (based on the lexicographical order of their
names).

Note that objects that don't do any computation by themselves but only
act as a container for other objects (e.g. a `NeuronGroup` which contains a
`StateUpdater`, a `Resetter` and a `Thresholder`), don't have any value for
``when``, but pass on the given values for ``dt`` and ``order`` to their
containing objects.

Every new `Network` starts a simulation at time 0; `Network.t` is a read-only
attribute, to go back to a previous moment in time (e.g. to do another trial
of a simulation with a new noise instantiation) use the mechanism described
below.

Note that while it is allowed to change the `dt` of an object between runs (e.g.
to simulate/monitor an initial phase with a bigger time step than a later
phase), this change has to be compatible with the internal representation of
clocks as an integer value (the number of elapsed time steps). For example, you
can simulate an object for 100ms with a time step of 0.1ms (i.e. for 1000 steps)
and then switch to a ``dt`` of 0.5ms, the time will then be internally
represented as 200 steps. You cannot, however, switch to a dt of 0.3ms, because
100ms are not an integer multiple of 0.3ms.
