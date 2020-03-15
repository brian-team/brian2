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

For C++ standalone mode, printing the progress to a file can be done similar to,

::
    stat = '''
    std::fstream fout; 
    fout.open("../report.txt", std::fstream::app); 
    fout << (int)(completed*100.) << "% completed" << std::endl; 
    fout.close();
    '''
    run(10 * second, report = stat)

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

Similarly, for C++ standalone mode, the same can be done something equivalent to,

::

    progressing = '''

    int tot_tiks = 50;
    std::cout<<"["<<std::flush;
    int iter = 0;

    while(iter < completed * tot_tiks)
    {
        std::cout<<"#"<<std::flush;
        iter += 1;
    }

    if(completed == 1.0)
    {
        std::cout<<"]"<<endl<<std::flush;
    }

    while(iter >= 0)
    {
        std::cout<<"\b"<<std::flush;
        iter -= 1;
    }
    '''
    
    run(1000 * second, report= progressing)
