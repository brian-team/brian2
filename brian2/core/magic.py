import weakref
import inspect
import itertools
import gc

from brian2.units.fundamentalunits import check_units
from brian2.units.allunits import second
from brian2.utils.logger import get_logger

from .network import Network
from .base import BrianObject, device_override    

__all__ = ['MagicNetwork', 'magic_network',
           'MagicError',
           'run', 'stop', 'collect', 'store', 'restore',
           'start_scope',
           ]

logger = get_logger(__name__)


def _get_contained_objects(obj):
    '''
    Helper function to recursively get all contained objects.

    Parameters
    ----------
    obj : `BrianObject`
        An object that (potentially) contains other objects, e.g. a
        `NeuronGroup` contains a `StateUpdater`, etc.

    Returns
    -------
    l : list of `BrianObject`
        A list of all the objects contained in `obj`
    '''
    l = []
    contained_objects = getattr(obj, 'contained_objects', [])
    l.extend(contained_objects)
    for contained_obj in contained_objects:
        l.extend(_get_contained_objects(contained_obj))

    return l


def get_objects_in_namespace(level):
    '''
    Get all the objects in the current namespace that derive from `BrianObject`.
    Used to determine the objects for the `MagicNetwork`.

    Parameters
    ----------
    level : int, optional
        How far to go back to get the locals/globals. Each function/method
        call should add ``1`` to this argument, functions/method with a
        decorator have to add ``2``.

    Returns
    -------
    objects : set
        A set with weak references to the `BrianObject`\ s in the namespace.
    '''
    # Get the locals and globals from the stack frame
    objects = set()
    frame = inspect.stack()[level + 1][0]
    for k, v in itertools.chain(frame.f_globals.iteritems(),
                                frame.f_locals.iteritems()):
        # We are only interested in numbers and functions, not in
        # everything else (classes, modules, etc.)
        if isinstance(v, BrianObject):
            objects.add(weakref.ref(v))
    del frame
    return objects


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
    
    All Brian objects that are visible at the point of the `run` call will be
    included in the network. This class is designed to work in the following
    two major use cases:
    
    1. You create a collection of Brian objects, and call `run` to run the
       simulation. Subsequently, you may call `run` again to run it again for
       a further duration. In this case, the `Network.t` time will start at 0
       and for the second call to `run` will continue from the end of the
       previous run.
       
    2. You have a loop in which at each iteration, you create some Brian
       objects and run a simulation using them. In this case, time is reset to
       0 for each call to `run`.
       
    In any other case, you will have to explicitly create a `Network` object
    yourself and call `Network.run` on this object. Brian has a built in
    system to guess which of the cases above applies and behave correctly.
    When it is not possible to safely guess which case you are in, it raises
    `MagicError`. The rules for this guessing system are explained below.
    
    If a simulation consists only of objects that have not been run, it will
    assume that you want to start a new simulation. If a simulation only
    consists of objects that have been simulated in the previous `run` call,
    it will continue that simulation at the previous time.

    If neither of these two situations apply, i.e., the network consists of a
    mix of previously run objects and new objects, an error will be raised.

    In these checks, "non-invalidating" objects (i.e. objects that have
    `BrianObject.invalidates_magic_network` set to ``False``) are ignored, e.g.
    creating new monitors is always possible.
    
    See Also
    --------
    Network, collect, run, stop, store, restore
    '''
    
    _already_created = False
    
    def __init__(self):
        if MagicNetwork._already_created:
            raise ValueError("There can be only one MagicNetwork.")
        MagicNetwork._already_created = True
        
        super(MagicNetwork, self).__init__(name='magicnetwork*')
        
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

    def _update_magic_objects(self, level):
        objects = collect(level+1)
        contained_objects = set()
        for obj in objects:
            for contained in _get_contained_objects(obj):
                contained_objects.add(contained)
        objects |= contained_objects

        # check whether we should restart time, continue time, or raise an
        # error
        some_known = False
        some_new = False
        for obj in objects:
            if obj._network == self.id:
                some_known = True  # we are continuing a previous run
            elif obj._network is None and obj.invalidates_magic_network:
                some_new = True
            # Note that the inclusion of objects that have been run as part of
            # other objects will lead to an error in `Network.before_run`, we
            # do not have to deal with this case here.

        if some_known and some_new:
            raise MagicError(('The magic network contains a mix of objects '
                              'that has been run before and new objects, Brian '
                              'does not know whether you want to start a new '
                              'simulation or continue an old one. Consider '
                              'explicitly creating a Network object. Also note '
                              'that you can find out which objects will be '
                              'included in a magic network with the '
                              'collect() function.'))
        elif some_new:  # all objects are new, start a new simulation
            # reset time
            self.t_ = 0.0
            # reset id -- think of this as a new Network
            self.assign_id()

        for obj in objects:
            if obj._network is None:
                obj._network = self.id

        self.objects[:] = objects
        logger.debug("Updated MagicNetwork to include {numobjs} objects "
                     "with names {names}".format(
                numobjs=len(self.objects),
                names=', '.join(obj.name for obj in self.objects)),
                name_suffix='magic_objects')

    def check_dependencies(self):
        all_ids = set([obj.id for obj in self.objects])
        for obj in self.objects:
            if not obj.active:
                continue  # object is already inactive, no need to check it
            for dependency in obj._dependencies:
                if not dependency in all_ids:
                    logger.warn(('"%s" has been included in the network but '
                                 'not the object on which it depends.'
                                 'Setting "%s" to inactive.') % (obj.name,
                                                                 obj.name),
                                name_suffix='dependency_warning')
                    obj.active = False
                    break

    def after_run(self):
        self.objects[:] = []
        gc.collect()  # Make sure that all unused objects are cleared

    def run(self, duration, report=None, report_period=10*second,
            namespace=None, profile=False, level=0):
        self._update_magic_objects(level=level+1)
        Network.run(self, duration, report=report, report_period=report_period,
                    namespace=namespace, profile=profile, level=level+1)

    def store(self, name='default', filename=None, level=0):
        '''
        See `Network.store`.
        '''
        self._update_magic_objects(level=level+1)
        super(MagicNetwork, self).store(name=name, filename=filename)
        self.objects[:] = []

    def restore(self, name='default', filename=None, level=0):
        '''
        See `Network.store`.
        '''
        self._update_magic_objects(level=level+1)
        super(MagicNetwork, self).restore(name=name, filename=filename)
        self.objects[:] = []

    def get_states(self, units=True, format='dict', subexpressions=False,
                   level=0):
        '''
        See `Network.get_states`.
        '''
        self._update_magic_objects(level=level+1)
        states = super(MagicNetwork, self).get_states(units, format,
                                                      subexpressions,
                                                      level=level+1)
        self.objects[:] = []
        return states

    def set_states(self, values, units=True, format='dict', level=0):
        '''
        See `Network.set_states`.
        '''
        self._update_magic_objects(level=level+1)
        super(MagicNetwork, self).set_states(values, units, format,
                                             level=level+1)
        self.objects[:] = []

    def __str__(self):
        return 'MagicNetwork()'
    __repr__ = __str__


#: Automatically constructed `MagicNetwork` of all Brian objects
magic_network = MagicNetwork()


def collect(level=0):
    '''
    Return the list of `BrianObject`\ s that will be simulated if `run` is
    called.

    Parameters
    ----------
    level : int, optional
        How much further up to go in the stack to find the objects. Needs
        only to be specified if `collect` is called as part of a function
        and should be increased by 1 for every level of nesting. Defaults to 0.

    Returns
    -------
    objects : set of `BrianObject`
        The objects that will be simulated.
    '''
    all_objects = set()
    for obj in get_objects_in_namespace(level=level+1):
        obj = obj()
        if obj.add_to_magic_network:
            gk = BrianObject._scope_current_key
            k = obj._scope_key
            if gk!=k:
                continue
            all_objects.add(obj)
    return all_objects


@check_units(duration=second, report_period=second)
def run(duration, report=None, report_period=10*second, namespace=None,
        profile=False, level=0):
    '''
    run(duration, report=None, report_period=10*second, namespace=None, level=0)
    
    Runs a simulation with all "visible" Brian objects for the given duration.
    Calls `collect` to gather all the objects, the simulation can
    be stopped by calling the global `stop` function.
    
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
    profile : bool, optional
        Whether to record profiling information (see `Network.profiling_info`).
        Defaults to ``False``.
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

    Network.run, MagicNetwork, collect, start_scope, stop

    Raises
    ------

    MagicError
        Error raised when it was not possible for Brian to safely guess the
        intended use. See `MagicNetwork` for more details.
    '''
    return magic_network.run(duration, report=report, report_period=report_period,
                             namespace=namespace, profile=profile, level=2+level)
run.__module__ = __name__


def store(name='default', filename=None):
    '''
    Store the state of the network and all included objects.

    Parameters
    ----------
    name : str, optional
        A name for the snapshot, if not specified uses ``'default'``.
    filename : str, optional
        A filename where the state should be stored. If not specified, the
        state will be stored in memory.

    See Also
    --------
    Network.store
    '''
    magic_network.store(name=name, filename=filename, level=1)


def restore(name='default', filename=None):
    '''
    Restore the state of the network and all included objects.

    Parameters
    ----------
    name : str, optional
        The name of the snapshot to restore, if not specified uses
        ``'default'``.
    filename : str, optional
        The name of the file from where the state should be restored. If
        not specified, it is expected that the state exist in memory
        (i.e. `Network.store` was previously called without the ``filename``
        argument).

    See Also
    --------
    Network.restore
    '''
    magic_network.restore(name=name, filename=filename, level=1)


def stop():
    '''
    Stops all running simulations.
    
    See Also
    --------
    
    Network.stop, run, reinit
    '''
    Network._globally_stopped = True


def start_scope():
    '''
    Starts a new scope for magic functions
    
    All objects created before this call will no longer be automatically
    included by the magic functions such as `run`.
    '''
    BrianObject._scope_current_key += 1
