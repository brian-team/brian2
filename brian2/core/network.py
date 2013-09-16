import weakref
import time

from brian2.utils.logger import get_logger
from brian2.core.names import Nameable
from brian2.core.base import BrianObject
from brian2.core.clocks import Clock
from brian2.units.fundamentalunits import check_units
from brian2.units.allunits import second 
from brian2.core.preferences import brian_prefs
from brian2.core.namespace import get_local_namespace

__all__ = ['Network']


logger = get_logger(__name__)


class Network(Nameable):
    '''
    Network(*objs, name='network*')
    
    The main simulation controller in Brian

    `Network` handles the running of a simulation. It contains a set of Brian
    objects that are added with `~Network.add`. The `~Network.run` method
    actually runs the simulation. The main run loop, determining which
    objects get called in what order is described in detail in the notes below.
    The objects in the `Network` are accesible via their names, e.g.
    `net['neurongroup']` would return the `NeuronGroup` with this name.
    
    Parameters
    ----------
    objs : (`BrianObject`, container), optional
        A list of objects to be added to the `Network` immediately, see
        `~Network.add`.
    name : str, optional
        An explicit name, if not specified gives an automatically generated name
    weak_references : bool, optional
        Whether to only store weak references to the objects (defaults to
        ``False``), i.e. other references to the objects need to exist to keep
        them alive. This is used by the magic system, otherwise it would keep
        all objects alive all the time.

    Notes
    -----
    
    The main run loop performs the following steps:
    
    1. Prepare the objects if necessary, see `~Network.prepare`.
    2. Determine the end time of the simulation as `~Network.t`+``duration``.
    3. Determine which set of clocks to update. This will be the clock with the
       smallest value of `~Clock.t`. If there are several with the same value,
       then all objects with these clocks will be updated simultaneously.
       Set `~Network.t` to the clock time.
    4. If the `~Clock.t` value of these clocks is past the end time of the
       simulation, stop running. If the `Network.stop` method or the
       `stop` function have been called, stop running. Set `~Network.t` to the
       end time of the simulation.
    5. For each object whose `~BrianObject.clock` is set to one of the clocks from the
       previous steps, call the `~BrianObject.update` method. This method will
       not be called if the `~BrianObject.active` flag is set to ``False``.
       The order in which the objects are called is described below.
    6. Increase `Clock.t` by `Clock.dt` for each of the clocks and return to
       step 2. 
    
    The order in which the objects are updated in step 4 is determined by
    the `Network.schedule` and the objects `~BrianObject.when` and
    `~BrianObject.order` attributes. The `~Network.schedule` is a list of
    string names. Each `~BrianObject.when` attribute should be one of these
    strings, and the objects will be updated in the order determined by the
    schedule. The default schedule is
    ``['start', 'groups', 'thresholds', 'synapses', 'resets', 'end']``. That
    means that all objects with ``when=='start'` will be updated first, then
    those with ``when=='groups'``, and so forth. If several objects have the
    same `~BrianObject.when` attribute, then the order is determined by the
    `~BrianObject.order` attribute (lower first).
    
    See Also
    --------
    
    MagicNetwork, run, stop
    '''

    def __init__(self, *objs, **kwds):
        #: The list of objects in the Network, should not normally be modified
        #: directly
        #:
        #: Stores references or `weakref.proxy` references to the objects
        #: (depending on `weak_references`)
        self.objects = []
        
        name = kwds.pop('name', 'network*')

        #: Whether the network only stores weak references to the objects
        self.weak_references = kwds.pop('weak_references', False)
        if kwds:
            raise TypeError("Only keyword arguments to Network are name "
                            "and weak_references")

        Nameable.__init__(self, name=name)

        for obj in objs:
            self.add(obj)
            
        #: Current time as a float
        self.t_ = 0.0   
     
    t = property(fget=lambda self: self.t_*second,
                 fset=lambda self, val: setattr(self, 't_', float(val)),
                 doc='''
                     Current simulation time in seconds (`Quantity`)
                     ''')

    _globally_stopped = False

    def __getitem__(self, item):
        if not isinstance(item, basestring):
            raise TypeError(('Need a name to access objects in a Network, '
                             'got {type} instead').format(type=type(item)))
        for obj in self.objects:
            if obj.name == item:
                return obj

        raise KeyError('No object with name "%s" found' % item)

    def __delitem__(self, key):
        if not isinstance(key, basestring):
            raise TypeError(('Need a name to access objects in a Network, '
                             'got {type} instead').format(type=type(key)))

        for obj in self.objects:
            if obj.name == key:
                self.remove(obj)
                return

        raise KeyError('No object with name "%s" found' % key)

    def __len__(self):
        return len(self.objects)

    def __iter__(self):
        return iter(self.objects)

    def add(self, *objs):
        """
        Add objects to the `Network`
        
        Parameters
        ----------
        
        objs : (`BrianObject`, container)
            The `BrianObject` or container of Brian objects to be added. Specify
            multiple objects, or lists (or other containers) of objects.
            Containers will be added recursively.
        """
        for obj in objs:
            if isinstance(obj, BrianObject):
                if self.weak_references and not isinstance(obj,
                                                           (weakref.ProxyType,
                                                            weakref.CallableProxyType)):
                    obj = weakref.proxy(obj)
                self.objects.append(obj)
                self.add(obj.contained_objects)
            else:
                try:
                    for o in obj:
                        self.add(o)
                except TypeError:
                    raise TypeError("Can only add objects of type BrianObject, "
                                    "or containers of such objects to Network")

    def remove(self, *objs):
        '''
        Remove an object or sequence of objects from a `Network`.
        
        Parameters
        ----------
        
        objs : (`BrianObject`, container)
            The `BrianObject` or container of Brian objects to be removed. Specify
            multiple objects, or lists (or other containers) of objects.
            Containers will be removed recursively.
        '''
        for obj in objs:
            if isinstance(obj, BrianObject):
                if self.weak_references and not isinstance(obj,
                                                           (weakref.ProxyType,
                                                            weakref.CallableProxyType)):
                    obj = weakref.proxy(obj)
                # note that weakref.proxy(obj) is weakref.proxy(obj) is True
                self.objects.remove(obj)
                self.remove(obj.contained_objects)
            else:
                try:
                    for o in obj:
                        self.remove(o)
                except TypeError:
                    raise TypeError("Can only remove objects of type "
                                    "BrianObject, or containers of such "
                                    "objects from Network")

    def reinit(self):
        '''
        Reinitialises all contained objects.
        
        Calls `BrianObject.reinit` on each object.
        '''
        for obj in self.objects:
            obj.reinit()
    
    def _get_schedule(self):
        if not hasattr(self, '_schedule'):
            self._schedule = ['start',
                              'groups',
                              'thresholds',
                              'synapses',
                              'resets',
                              'end',
                              ]
        return self._schedule            
    
    def _set_schedule(self, schedule):
        self._schedule = schedule
        logger.debug("Set network {self.name} schedule to "
                     "{self._schedule}".format(self=self),
                     "_set_schedule")
    
    schedule = property(fget=_get_schedule,
                        fset=_set_schedule,
                        doc='''
        List of ``when`` slots in the order they will be updated, can be modified.
        
        See notes on scheduling in `Network`. Note that additional ``when``
        slots can be added, but the schedule should contain at least all of the
        names in the default schedule:
        ``['start', 'groups', 'thresholds', 'synapses', 'resets', 'end']``.
        ''')
    
    def _sort_objects(self):
        '''
        Sorts the objects in the order defined by the schedule.
        
        Objects are sorted first by their ``when`` attribute, and secondly
        by the ``order`` attribute. The order of the ``when`` attribute is
        defined by the ``schedule``.
        '''
        when_to_int = dict((when, i) for i, when in enumerate(self.schedule))
        self.objects.sort(key=lambda obj: (when_to_int[obj.when], obj.order))
    
    def before_run(self, namespace):
        '''
        Prepares the `Network` for a run.
        
        Objects in the `Network` are sorted into the correct running order, and
        their `BrianObject.before_run` methods are called.
        '''                
        brian_prefs.check_all_validated()

        self._clocks = set(obj.clock for obj in self.objects)
        
        self._stopped = False
        Network._globally_stopped = False
        
        self._sort_objects()

        logger.debug("Preparing network {self.name} with {numobj} "
                     "objects: {objnames}".format(self=self,
                        numobj=len(self.objects),
                        objnames=', '.join(obj.name for obj in self.objects)),
                     "before_run")
        
        for obj in self.objects:
            obj.before_run(namespace)

        logger.debug("Network {self.name} has {num} "
                     "clocks: {clocknames}".format(self=self,
                        num=len(self._clocks),
                        clocknames=', '.join(obj.name for obj in self._clocks)),
                     "before_run")
    
    def after_run(self):
        for obj in self.objects:
            obj.after_run()
        
    def _nextclocks(self):
        minclock = min(self._clocks, key=lambda c: c.t_)
        curclocks = set(clock for clock in self._clocks if
                        (clock.t_ == minclock.t_ or
                         abs(clock.t_ - minclock.t_)<Clock.epsilon))
        return minclock, curclocks
    
    @check_units(duration=second, report_period=second)
    def run(self, duration, report=None, report_period=60*second,
            namespace=None, level=0):
        '''
        run(duration, report=None, report_period=60*second)
        
        Runs the simulation for the given duration.
        
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
        namespace : dict-like, optional
            A namespace in which objects which do not define their own
            namespace will be run. If not namespace is given, the locals and
            globals around the run function will be used. 
        level : int, optional
            How deep to go down the stack frame to look for the locals/global
            (see `namespace` argument). Only used by run functions that call
            this run function, e.g. `MagicNetwork.run` to adjust for the
            additional nesting.
        Notes
        -----
        
        The simulation can be stopped by calling `Network.stop` or the
        global `stop` function.
        '''
        
        if namespace is not None:
            self.before_run(('explicit-run-namespace', namespace))
        else:
            namespace = get_local_namespace(2 + level)
            self.before_run(('implicit-run-namespace', namespace))

        if len(self.objects)==0:
            return # TODO: raise an error? warning?

        t_end = self.t+duration
        for clock in self._clocks:
            clock.set_interval(self.t, t_end)
            
        # TODO: progress reporting stuff
        
        # Find the first clock to be updated (see note below)
        clock, curclocks = self._nextclocks()
        if report is not None:
            start = current = time.time()
            next_report_time = start + 10

        while clock.running and not self._stopped and not Network._globally_stopped:
            # update the network time to this clocks time
            self.t_ = clock.t_
            if report is not None:
                current = time.time()
                if current > next_report_time:
                    report_msg = '{t} simulated ({percent}%), estimated {remaining} s remaining.'
                    remaining = int(round((current - start)/self.t*(duration-self.t)))
                    print report_msg.format(t=self.t, percent=int(round(100*self.t/duration)),
                                            remaining=remaining)
                    next_report_time = current + 10
                # update the objects with this clock
            for obj in self.objects:
                if obj.clock in curclocks and obj.active:
                    for updater in obj.updaters:
                        updater.run()
            # tick the clock forward one time step
            for c in curclocks:
                c.tick()
            # find the next clocks to be updated. The < operator for Clock
            # determines that the first clock to be updated should be the one
            # with the smallest t value, unless there are several with the 
            # same t value in which case we update all of them
            clock, curclocks = self._nextclocks()

        self.t = t_end

        if report is not None:
            print 'Took ', current-start, 's in total.'
        self.after_run()
        
    def stop(self):
        '''
        Stops the network from running, this is reset the next time `Network.run` is called.
        '''
        self._stopped = True

    def __repr__(self):
        return '<%s at time t=%s, containing objects: %s>' % (self.__class__.__name__,
                                                              str(self.t),
                                                              ', '.join((obj.__repr__() for obj in self.objects)))