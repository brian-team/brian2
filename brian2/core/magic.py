import sys
import weakref

from brian2.units.fundamentalunits import check_units
from brian2.units.allunits import second
from brian2.utils.logger import get_logger
from brian2.core.network import Network
from brian2.core.base import BrianObject
from brian2.utils.proxy import get_proxy_count

__all__ = ['MagicNetwork', 'magic_network',
           'MagicError',
           'run', 'reinit', 'stop',
           ]

logger = get_logger(__name__)


class MagicError(Exception):
    '''
    Error that is raised when something goes wrong in `MagicNetwork`
    
    See notes to `MagicNetwork` for more details.
    '''
    pass


class MagicNetwork(Network):
    '''
    `Network` that automatically adds all Brian objects
    
    In order to avoid bugs, this class will occasionally raise
    `MagicError` when the intent of the user is not clear. See the notes
    below for more details on this point. If you persistently see this
    error, then Brian is not able to safely guess what you intend to do, and
    you should use a `Network` object and call `Network.run` explicitly.
    
    Note that this class cannot be instantiated by the user, there can be only
    one instance `magic_network` of `MagicNetwork`.

    Notes
    -----
    
    All Brian objects that have not been removed by the `clear` function will
    be included in the network. This class is designed to work in the following
    two major use cases:
    
    1. You create a collection of Brian objects, and call `run` to run the
       simulation. Subsequently, you may call `run` again to run it again for
       a further duration. In this case, the `Network.t` time will start at 0
       and for the second call to `run` will continue from the end of the
       previous run. To reset time to 0 between runs, write
       ``magic_network.t = 0*second``.
       
    2. You have a loop in which at each iteration, you create some Brian
       objects and run a simulation using them. In this case, time is reset to
       0 for each call to `run`.
       
    In any other case, you will have to explicitly create a `Network` object
    yourself and call `Network.run` on this object. Brian has a built in
    system to guess which of the cases above applies and behave correctly.
    When it is not possible to safely guess which case you are in, it raises
    `MagicError`. The rules for this guessing system are explained below.
    
    The rule is essentially this: if you are running a network consisting of
    entirely new objects compared to the previous run, then time will be reset
    to 0 and no error is raised. If you are running a network consisting only
    of objects that existed on the previous run, time continues from the end
    of the previous run and no error is raised. If the set of objects is
    different but has some overlap, an error is raised. So, for example,
    creating a new `NeuronGroup` and calling `run` will raise an error. The
    reason for this raising an error is that (a) Brian cannot guess the
    intent of the user, and doesn't know whether to reset time to 0 or not.
    (b) Occasionally, this indicates a problem that references to previously
    existing Brian objects - from a previous iteration of a loop for example -
    still exist. Normally, the user will not want these to be included in the
    run, and they still exist either because Python garbage collection wasn't
    able to remove all the references, or because the user is storing some
    objects to retain their data. In this case, Brian has no way to know
    which objects should or shouldn't be included in the run and so raises an
    error. In this case, you should use a `Network` object explicitly.
    
    There is a slight subtlety to the rules above: adding or removing some
    types of Brian object will not cause an error to be raised. All Brian
    objects have a `~BrianObject.invalidates_magic_network` attribute - if this
    flag is set to ``False`` then adding it will not cause an error to be
    raised. You can check this attribute on each object, but the basic rule is
    that the flag will be set to ``True`` for "stand-alone" Brian objects which
    can be run on their own, e.g. `NeuronGroup` and ``False`` for objects which
    require the existence of another object such as `Synapses` or
    `SpikeMonitor`.
    
    See Also
    --------
    
    Network, run, reinit, stop, clear
    '''
    
    _already_created = False
    
    def __init__(self):
        if MagicNetwork._already_created:
            raise ValueError("There can be only one MagicNetwork.")
        MagicNetwork._already_created = True
        
        super(MagicNetwork, self).__init__(name='magicnetwork*',
                                           weak_references=True)
        
        self._previous_refs = set()
        
    def add(self, *objs):
        '''
        You cannot add objects directly to `MagicNetwork`
        '''
        raise MagicError("Cannot directly modify MagicNetwork")

    def remove(self, *objs):
        '''
        You cannot remove objects directly from `MagicNetwork`
        '''
        raise MagicError("Cannot directly modify MagicNetwork")
    
    def _update_magic_objects(self):
        # Go through all the objects and ignore those that are only referred to
        # by Proxy objects (e.g. because a Monitor holds a reference to them)
        valid_refs = set()
        all_objects = set()
        for obj in BrianObject.__instances__():
            obj = obj()
            proxycount = get_proxy_count(obj)
            # subtract 1 from refcount for refcount arg
            # subtract 1 from refcount for refcount in this loop
            refcount = sys.getrefcount(obj)-2
            if refcount != proxycount:
                all_objects.add(obj)
                if obj.invalidates_magic_network:
                    valid_refs.add(weakref.ref(obj))

        # check whether we should restart time, continue time, or raise an
        # error
        inter = valid_refs.intersection(self._previous_refs)

        if len(inter)==0:
            # reset time
            self.t = 0*second
        elif len(self._previous_refs)==len(valid_refs):
            # continue time
            pass
        else:
            raise MagicError("Brian cannot guess what you intend to do here, see docs for MagicNetwork for details")
        self._previous_refs = valid_refs
        self.objects[:] = list(all_objects)
        logger.debug("Updated MagicNetwork to include {numobjs} objects "
                     "with names {names}".format(
                        numobjs=len(self.objects),
                        names=', '.join(obj.name for obj in self.objects)))

    def before_run(self, run_namespace=None, level=0):
        self._update_magic_objects()
        Network.before_run(self, run_namespace, level=level+1)

    def after_run(self):
        self.objects[:] = []

    def reinit(self):
        '''
        See `Network.reinit`.
        '''
        self._update_magic_objects()
        super(MagicNetwork, self).reinit()
        
    def __str__(self):
        return 'MagicNetwork()'
    __repr__ = __str__


#: Automatically constructed `MagicNetwork` of all Brian objects
magic_network = MagicNetwork()


@check_units(duration=second, report_period=second)
def run(duration, report=None, report_period=60*second, namespace=None,
        level=0):
    '''
    run(duration, report=None, report_period=60*second, namespace=None)
    
    Runs a simulation with all Brian objects for the given duration.
    Objects can be reinitialised using `reinit` and
    the simulation can be stopped by calling the global `stop` function.
    
    In order to avoid bugs, this function will occasionally raise
    `MagicError` when the intent of the user is not clear. See the notes to
    `MagicNetwork` for more details on this point. If you persistently see this
    error, then Brian is not able to safely guess what you intend to do, and
    you should use a `Network` object and call `Network.run` explicitly.
    
    Parameters
    ----------
    
    duration : `Quantity`
        The amount of simulation time to run for. If the network consists of
        new objects since the last time `run` was called, the start time will
        be reset to 0. If `run` is called twice or more without changing the
        set of objects, the second and subsequent runs will start from the
        end time of the previous run. To explicitly reset the time to 0,
        do ``magic_network.t = 0*second``.
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
    namespace : dict-like, optional
        A namespace in which objects which do not define their own
        namespace will be run. If not namespace is given, the locals and
        globals around the run function will be used.
    level : int, optional
        How deep to go down the stack frame to look for the locals/global
        (see `namespace` argument). Only necessary under particular
        circumstances, e.g. when calling the run function as part of a
        function call or lambda expression. This is used in tests, e.g.:
        ``assert_raises(MagicError, lambda: run(1*ms, level=3))``.

    See Also
    --------
    
    Network.run, MagicNetwork, reinit, stop, clear
    
    Raises
    ------
    
    MagicError
        Error raised when it was not possible for Brian to safely guess the
        intended use. See `MagicNetwork` for more details.
    '''
    magic_network.run(duration, report=report, report_period=report_period,
                      namespace=namespace, level=2+level)
run.__module__ = __name__

def reinit():
    '''
    Reinitialises all Brian objects.
    
    This function works similarly to `run`, see the documentation for that
    function for more details.
    
    See Also
    --------
    
    Network.reinit, run, stop, clear
    
    Raises
    ------
    
    MagicError
        Error raised when it was not possible for Brian to safely guess the
        intended use. See `MagicNetwork` for more details.
    '''
    magic_network.reinit()


def stop():
    '''
    Stops all running simulations.
    
    See Also
    --------
    
    Network.stop, run, reinit
    '''
    Network._globally_stopped = True
