import weakref

from brian2.utils.logger import get_logger
from brian2.core.names import Nameable
from brian2.core.base import BrianObject
from brian2.core.clocks import Clock
from brian2.units.fundamentalunits import check_units
from brian2.units.allunits import second 
from brian2.core.preferences import brian_prefs

__all__ = ['Network']

logger = get_logger(__name__)


class Network(Nameable):
    '''
    Network(*objs, name=None)
    
    The main simulation controller in Brian

    `Network` handles the running of a simulation. It contains a set of Brian
    objects that are added with `~Network.add`. The `~Network.run` method
    actually runs the simulation. The main run loop, determining which
    objects get called in what order is described in detail in the notes below.
    
    Parameters
    ----------
    objs : (`BrianObject`, container), optional
        A list of objects to be added to the `Network` immediately, see
        `~Network.add`.
    name : (str, None), optional
        An explicit name, if not specified gives an automatically generated name
        
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
    
    basename = 'network'
    name = Nameable.name
    
    def __init__(self, *objs, **kwds):
        #: The list of objects in the Network, should not normally be modified directly
        #:
        #: Stores `weakref.proxy` references to the objects.
        self.objects = []
        
        name = kwds.pop('name', None)
        if kwds:
            raise TypeError("Only keyword argument to Network is name")
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
                #self.objects.append(obj)
                if not isinstance(obj, (weakref.ProxyType, weakref.CallableProxyType)):
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
                if not isinstance(obj, (weakref.ProxyType, weakref.CallableProxyType)):
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
    
    def pre_run(self, namespace):
        '''
        Prepares the `Network` for a run.
        
        Objects in the `Network` are sorted into the correct running order, and
        their `BrianObject.pre_run` methods are called.
        '''                
        brian_prefs.check_all_validated()
        
        if len(self.objects)==0:
            return # TODO: raise an error? warning?
        
        self._stopped = False
        Network._globally_stopped = False
        
        self._sort_objects()

        logger.debug("Preparing network {self.name} with {numobj} "
                     "objects: {objnames}".format(self=self,
                        numobj=len(self.objects),
                        objnames=', '.join(obj.name for obj in self.objects)),
                     "pre_run")
        
        for obj in self.objects:
            obj.pre_run(namespace)
        
        self._clocks = set(obj.clock for obj in self.objects)

        logger.debug("Network {self.name} has {num} "
                     "clocks: {clocknames}".format(self=self,
                        num=len(self._clocks),
                        clocknames=', '.join(obj.name for obj in self._clocks)),
                     "pre_run")
    
    def post_run(self):
        for obj in self.objects:
            obj.post_run()        
        
    def _nextclocks(self):
        minclock = min(self._clocks, key=lambda c: c.t_)
        curclocks = set(clock for clock in self._clocks if
                        (clock.t_ == minclock.t_ or
                         abs(clock.t_ - minclock.t_)<Clock.epsilon))
        return minclock, curclocks
    
    @check_units(duration=second, report_period=second)
    def run(self, duration, report=None, report_period=60*second):
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
            
        Notes
        -----
        
        The simulation can be stopped by calling `Network.stop` or the
        global `stop` function.
        '''
        
        # FIXME: local/global Namespace 
        self.pre_run({})

        t_end = self.t+duration
        for clock in self._clocks:
            clock.set_interval(self.t, t_end)
            
        # TODO: progress reporting stuff
        
        # Find the first clock to be updated (see note below)
        clock, curclocks = self._nextclocks()
        while clock.running and not self._stopped and not Network._globally_stopped:
            # update the network time to this clocks time
            self.t = clock.t
            # update the objects with this clock
            for obj in self.objects:
                if obj.clock in curclocks and obj.active:
                    obj.update()
            # tick the clock forward one time step
            for c in curclocks:
                c.tick()
            # find the next clocks to be updated. The < operator for Clock
            # determines that the first clock to be updated should be the one
            # with the smallest t value, unless there are several with the 
            # same t value in which case we update all of them
            clock, curclocks = self._nextclocks()
            
        self.t = t_end
        
        self.post_run()
        
    def stop(self):
        '''
        Stops the network from running, this is reset the next time `Network.run` is called.
        '''
        self._stopped = True
