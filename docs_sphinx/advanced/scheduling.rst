Custom progress reporting
=========================

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
        def __init__(self, toolbar_width=40):
            self.toolbar_width = toolbar_width
            self.ticks = 0

        def __call__(self, elapsed, complete, start, duration):
            if complete == 0.0:
                # setup toolbar
                sys.stdout.write("[%s]" % (" " * self.toolbar_width))
                sys.stdout.flush()
                sys.stdout.write("\b" * (self.toolbar_width + 1)) # return to start of line, after '['
            else:
                ticks_needed = int(round(complete * self.toolbar_width))
                if self.ticks < ticks_needed:
                    sys.stdout.write("-" * (ticks_needed-self.ticks))
                    sys.stdout.flush()
                    self.ticks = ticks_needed
            if complete == 1.0:
                sys.stdout.write("\n")

    net.run(duration, report=ProgressBar(), report_period=1*second)


**"Standalone Mode" Text based progress bar on console**

This needs a "normal" Linux console, i.e. it might not work in an integrated
console in an IDE.

Adapted from https://stackoverflow.com/questions/14539867/how-to-display-a-progress-indicator-in-pure-c-c-cout-printf

::

    from brian2 import *
    set_device('cpp_standalone', directory='STDP_standalone')

    str = '''
        int remaining = (int)((1-completed)/completed*elapsed+0.5);
        if (completed == 0.0)
        {
            std::cout << "Starting simulation at t=" << start << " s for duration " << duration << " s"<<std::flush;
        }
        else
        {
            int barWidth = 70;
            std::cout << "\\r[";
            int pos = barWidth * completed;
            for (int i = 0; i < barWidth; ++i) {
                    if (i < pos) std::cout << "=";
                    else if (i == pos) std::cout << ">";
                    else std::cout << " ";
            }
            std::cout << "] " << int(completed * 100.0) << "% completed. | "<<int(remaining) <<"s remaining"<<std::flush;
        }
    '''
    run(100*second, report=str)
