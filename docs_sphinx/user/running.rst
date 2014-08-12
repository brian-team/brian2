Running a simulation
====================

To run a simulation, one either constructs a new `Network` object and calls its
`Network.run` method, or uses the "magic" system and a plain `run` call,
collecting all the objects in the current namespace.

Magic networks
--------------
TODO

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