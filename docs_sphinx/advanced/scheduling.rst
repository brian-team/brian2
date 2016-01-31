Scheduling and custom progress reporting
========================================

Scheduling
----------
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
'resets', 'end']``. In addition to the names provided in the schedule, names
such as ``before_thresholds`` or ``after_synapses`` can be used that are
understood as slots in the respective positions. The default
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

.. _custom_progress_reporting:

Progress reporting
------------------
For custom progress reporting (e.g. graphical output, writing to a file, etc.),
the ``report`` keyword accepts a callable (i.e. a function or an object with a
``__call__`` method) that will be called with four parameters:

* ``elapsed``: the total (real) time since the start of the run
* ``completed``: the fraction of the total simulation that is completed,
  i.e. a value between 0 and 1
* ``start``: The start of the simulation (in biological time)
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

        def __call__(self, elapsed, complete, start, duration):
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
