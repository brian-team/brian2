Running a simulation
====================

To run a simulation, one either constructs a new `Network` object and calls its
`Network.run` method, or uses the "magic" system and a plain `run` call,
collecting all the objects in the current namespace.

Note that Brian has several different ways of running the actual computations,
and choosing the right one can make orders of magnitude of difference in
terms of simplicity and efficiency. See :doc:`computation` for more details.

Magic networks
--------------
In most straightforward simulations, you do not have to explicitly create a
`Network` object but instead can simply call `run` to run a simulation. This is
what is called the "magic" system, because Brian figures out automatically what
you want to do.

When calling `run`, Brian runs the `collect` function to gather all the objects
in the current context. It will include all the objects that are "visible", i.e.
that you could refer to with an explicit name::

  G = NeuronGroup(10, 'dv/dt = -v / tau : volt')
  S = Synapses(G, G, model='w:1', pre='v+=w', connect='i!=j')
  mon = SpikeMonitor(G)

  run(10*ms)  # will include G, S, mon

Note that it will not automatically include objects that are "hidden" in
containers, e.g. if you store several monitors in a list. Use an explicit
`Network` object in this case. It might be convenient to use the `collect`
function when creating the `Network` object in that case::

    G = NeuronGroup(10, 'dv/dt = -v / tau : volt')
    S = Synapses(G, G, model='w:1', pre='v+=w', connect='i!=j')
    monitors = [SpikeMonitor(G), StateMonitor(G, 'v', record=True)]

    # a simple run would not include the monitors
    net = Network(collect())  # automatically include G and S
    net.add(monitors)  # manually add the monitors

When you use more than a single `run` statement, the magic system tries to
detect which of the following two situations applies:

1. You want to continue a previous simulation
2. You want to start a new simulation

For this, it uses the following heuristic: if a simulation consists only of
objects that have not been run, it will start a new simulation starting at
time 0 (corresponding to the creation of a new `Network` object). If a
simulation only consists of objects that have been simulated in the previous
`run` call, it will continue that simulation at the previous time.

If neither of these two situations apply, i.e., the network consists of a mix
of previously run objects and new objects, an error will be raised. If this is
not a mistake but intended (e.g. when a new input source and synapses should be
added to a network at a later stage), use an explicit `Network` object.

In these checks, "non-invalidating" objects (i.e. objects that have
`BrianObject.invalidates_magic_network` set to ``False``) are ignored, e.g.
creating new monitors is always possible.

Progress reporting
------------------
Especially for long simulations it is useful to get some feedback about the
progress of the simulation. Brian offers a few built-in options and an
extensible system to report the progress of the simulation. In the `Network.run`
or `run` call, two arguments determine the output: ``report`` and
``report_period``. When ``report`` is set to ``'text'`` or ``'stdout'``, the
progress will be printed to the standard output, when it is set to ``'stderr'``,
it will be printed to "standard error". There will be output at the start and
the end of the run, and during the run in ``report_period`` intervals. For
custom progress reporting (e.g. graphical output, writing to a file, etc.), the
``report`` keyword accepts a callable (i.e. a function or an object with a
``__call__`` method) that will be called with three parameters:

* ``elapsed``: the total (real) time since the start of the run
* ``completed``: the fraction of the total simulation that is completed,
  i.e. a value between 0 and 1
* ``duration``: the total duration (in biological time) of the simulation

The function will be called every ``report_period`` during the simulation, but
also at the beginning and end with ``completed`` equal to 0.0 and 1.0,
respectively.

For the C++ standalone mode, the same standard options are available. It is
also possible to implement custom progress reporting by directly passing the
code (as a multi-line string) to the ``report`` argument. This code will be
filled into a progress report function template, it should therefore only
contain a function body. The simplest use of this might look like::

    net.run(duration, report='std::cout << (int)(completed*100.) << "% completed" << std::endl;')



Examples of custom reporting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Progress printed to a file**
::

    from brian2.core.network import TextReport
    report_file = open('report.txt', 'w')
    file_reporter = TextReport(report_file)
    net.run(duration, report=file_reporter)
    report_file.close()

**"Graphical" output on the console**

This needs a "normal" Linux console, i.e. it might not work in an integrated
console in an IDE.

Adapted from http://stackoverflow.com/questions/3160699/python-progress-bar

::

    import sys

    class ProgressBar(object):
        def __init__(self, toolbar_width):
            self.toolbar_width = toolbar_width
            self.ticks = 0

        def __call__(self, elapsed, complete, duration):
            if complete == 0.0:
                # setup toolbar
                sys.stdout.write("[%s]" % (" " * self.toolbar_width))
                sys.stdout.flush()
                sys.stdout.write("\b" * (self.toolbar_width + 1)) # return to start of line, after '['
            else:
                ticks_needed = int(round(complete * 40))
                if self.ticks < ticks_needed:
                    sys.stdout.write("-" * (ticks_needed-self.ticks))
                    sys.stdout.flush()
                    self.ticks = ticks_needed
            if complete == 1.0:
                sys.stdout.write("\n")

    net.run(duration, report=progress_bar, report_period=1*second)

Scheduling
----------

Every simulated object in Brian has three attributes that can be specified at
object creation time: ``dt``, ``when``, and ``order``. The time step of the
simulation is determined by ``dt``, if it is specified, or otherwise by
``defaultclock.dt``. Changing this will therefore change the ``dt`` of
all objects that don't specify one.

During a single time step, objects are updated in an order according first
to their ``when``
argument's position in the schedule.  This schedule is determined by
`Network.schedule` which is a list of strings, determining "execution slots" and
their order. It defaults to: ``['start', 'groups', 'thresholds', 'synapses',
'resets', 'end']``. The default
for the ``when`` attribute is a sensible value for most objects (resets will
happen in the ``reset`` slot, etc.) but sometimes it make sense to change it,
e.g. if one would like a `StateMonitor`, which by default records in the
``end`` slot, to record the membrane potential before a reset is applied
(otherwise no threshold crossings will be observed in the membrane potential
traces).

Finally, if during a time step two objects fall in the same execution
slot, they will be updated in ascending order according to their
``order`` attribute, an integer number defaulting to 0. If two objects have
the same ``when`` and ``order`` attribute then they will be updated in an
arbitrary but reproducible order (based on the lexicographical order of their
names).

Every new `Network` starts a simulation at time 0; `Network.t` is a read-only
attribute, to go back to a previous moment in time (e.g. to do another trial
of a simulation with a new noise instantiation) use the mechanism described
below.

For more details, including finer control over the scheduling of operations
and changing the value of ``dt`` between runs see
:doc:`../advanced/scheduling`.

Continuing/repeating simulations
--------------------------------

To store the current state of a network, including the time of the simulation,
internal variables like triggered but not yet delivered spikes, etc., call
`Network.store` which will store the state of all the objects
in the network (use a plain `store` if you are using the magic system). You
can store more than one snapshot of a system by providing a name for the
snapshot; if `Network.store` is called without a specified name,
``'default'`` is used as the name. To restore a network's state, use
`Network.restore`.

The following simple example shows how this system can be used to run several
trials of an experiment::

    # set up the network
    G = NeuronGroup(...)
    S = Synapses(...)
    G.v = ...
    S.connect(...)
    S.w = ...
    spike_monitor = SpikeMonitor(G)
    # Snapshot the state
    store()

    # Run the trials
    spike_counts = []
    for trial in range(3):
        restore()  # Restore the initial state
        run(...)
        # store the results
        spike_counts.append(spike_monitor.count)

The following schematic shows how multiple snapshots can be used to run a
network with a separate "train" and "test" phase. After training, the test is
run several times based on the trained network. The whole process of training
and testing is repeated several times as well::

    # set up the network
    G = NeuronGroup(..., '''...
                         test_input : amp
                         ...''')
    S = Synapses(..., '''...
                         plastic : boolean (shared)
                         ...''')
    G.v = ...
    S.connect(...)
    S.w = ...

    # First snapshot at t=0
    store('initialized')

    # Run 3 complete trials
    for trial in range(3):
        # Simulate training phase
        restore('initialized')
        S.plastic = True
        run(...)

        # Snapshot after learning
        store('after_learning')

        # Run 5 tests after the training
        for test_number in range(5):
            restore('after_learning')
            S.plastic = False  # switch plasticity off
            G.test_input = test_inputs[test_number]
            # monitor the activity now
            spike_mon = SpikeMonitor(G)
            run(...)
            # Do something with the result
            # ...

Note that `Network.run`, `Network.store` and `Network.restore` (or `run`,
`store`, `restore`) are the only way of affecting the time of the clocks. In
contrast to Brian1, it is no longer necessary (nor possible) to directly set
the time of the clocks or call a ``reinit`` function.
