from ..units import check_units, second
from ..base import BrianObject, get_instances
from .network import Network
import network

__all__ = ['MagicNetwork',
           'run', 'reinit', 'stop',
           ]


class MagicNetwork(Network):
    '''
    `Network` that automatically adds all Brian objects
    
    Notes
    -----
    
    All Brian objects that have not been removed by the `clear` function will
    be included. The time `~Network.t` will be set to the minimal value over
    all the clocks of objects added at initialisation.
    
    See Also
    --------
    
    Network, run, reinit, stop, clear
    '''
    def __init__(self):
        super(MagicNetwork, self).__init__()
        self.add(get_instances(BrianObject))
        minclock = min(set(obj.clock for obj in self.objects))
        self.t = minclock.t


@check_units(duration=second, report_period=second)
def run(duration, report=None, report_period=60*second):
    '''
    run(duration, report=None, report_period=60*second)
    
    Runs a simulation with all Brian objects for the given duration.
    
    Parameters
    ----------
    
    duration : `Quantity`
        The amount of simulation time to run for.
    report : {None, 'stdout', 'stderr', 'graphical', function}, optional
        How to report the progress of the simulation. If None, do not
        report progress. If stdout or stderr is specified, print the
        progress to stdout or stderr. If graphical, Tkinter is used to
        show a graphical progress bar. Alternatively, you can specify
        a callback ``function(elapsed, complete)`` which will be passed
        the amount of time elapsed (in seconds) and the fraction complete
        from 0 to 1.
    report_period : `Quantity`
        How frequently (in real time) to report progress.
        
    Notes
    -----
    
    The simulation `Network` will include all defined Brian objects that have
    not been removed by the `clear` function. The start time of the simulation
    will be the minimum time of all the clocks of the objects found. This means
    that in certain unusual circumstances several calls to `run` can lead to
    unexpected time values. For
    example, if two clocks are present with dt=3*ms and dt=5*ms then a call to
    ``run(4*ms)`` followed by ``run(4*ms)`` will run for the time interval
    ``[0*ms, 4*ms)`` for the first call, and then ``[5*ms, 9*ms]``. This is
    because at the end of the first run the first clock time will be set to
    ``6*ms`` and the second to ``5*ms``, so the start time will be taken to be
    ``5*ms``. To fix this problem, either run for durations which are divisible
    by the smallest `~Clock.dt`, or use explicitly `MagicNetwork` or `Network`
    which remember times between runs. 

    The simulation can be stopped by calling the global :func:`stop` function.
    
    See Also
    --------
    
    Network.run, MagicNetwork, reinit, stop, clear
    '''
    net = MagicNetwork()
    net.run(duration, report=report, report_period=report_period)


def reinit():
    '''
    Reinitialises all Brian objects.
    
    See Also
    --------
    
    Network.reinit, MagicNetwork, run, stop, clear
    '''
    net = MagicNetwork()
    net.reinit()


def stop():
    '''
    Stops all running simulations.
    
    See Also
    --------
    
    Network.stop, MagicNetwork, run, reinit
    '''
    network.globally_stopped = True
