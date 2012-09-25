from ..units import check_units, second
from ..base import BrianObject, get_instances

__all__ = ['Network',
           ]

globally_stopped = False

class Network(object):
    '''
    TODO
    
    See Also
    --------
    
    MagicNetwork, run
    '''
    def __init__(self, *args):
        #: The list of objects in the Network
        self.objects = []
        
        self._prepared = False

        for obj in args:
            self.add(obj)            

    def add(self, *objs):
        """
        Add objects to the network
        
        Parameters
        ----------
        
        objs : (BrianObject, container)
            The BrianObject or container of Brian objects to be added. Specify
            multiple objects, or lists (or other containers) of objects.
            Containers will be added recursively.
        """
        self._prepared = False
        for obj in objs:
            if isinstance(obj, BrianObject):
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
        Remove an object or sequence of objects from a Network.
        
        Parameters
        ----------
        
        objs : (BrianObject, container)
            The BrianObject or container of Brian objects to be removed. Specify
            multiple objects, or lists (or other containers) of objects.
            Containers will be removed recursively.
        '''
        self._prepared = False
        for obj in objs:
            if isinstance(obj, BrianObject):
                self.objects = [o for o in self.objects if o is not obj]
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
        self._prepared = False
        self._schedule = schedule
    
    schedule = property(fget=_get_schedule,
                        fset=_set_schedule,
                        doc='''
        List of ``when`` slots in the order they will be updated.
        
        See notes on scheduling in Network.
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
    
    def prepare(self):
        '''
        Prepares the Network, including sorting and preparing objects.
        
        Objects in the Network are sorted into the correct running order, and
        their :meth:`BrianObject.prepare` methods are called.
        '''
        self._sort_objects()
        
        for obj in self.objects:
            obj.prepare()
        
        self._clocks = set(obj.clock for obj in self.objects)
            
        self._prepared = True
    
    @check_units(duration=second, report_period=second)
    def run(self, duration, report=None, report_period=60*second):
        '''
        Runs the simulation for the given duration.
        
        Parameters
        ----------
        
        duration : Quantity
            The amount of simulation time to run for.
        report : {None, 'stdout', 'stderr', 'graphical', function}, optional
            How to report the progress of the simulation. If None, do not
            report progress. If stdout or stderr is specified, print the
            progress to stdout or stderr. If graphical, Tkinter is used to
            show a graphical progress bar. Alternatively, you can specify
            a callback ``function(elapsed, complete)`` which will be passed
            the amount of time elapsed (in seconds) and the fraction complete
            from 0 to 1.
        report_period : Quantity
            How frequently (in real time) to report progress.
            
        Notes
        -----
        
        The simulation can be stopped by calling :meth:`Network.stop` or the
        global :func:`stop` function.
        '''
        global globally_stopped
        
        if len(self.objects)==0:
            return # TODO: raise an error? warning?
        
        self._stopped = False
        globally_stopped = False
        
        if not self._prepared:
            self.prepare()
            
        for clock in self._clocks:
            clock.set_duration(duration)
            
        # TODO: progress reporting stuff
        
        # Find the first clock to be updated (see note below)
        clock = min(self._clocks)
        while clock.running and not self._stopped and not globally_stopped:
            # update the objects with this clock
            for obj in self.objects:
                if obj.clock is clock:
                    obj.update()
            # tick the clock forward one time step
            clock.tick()
            # find the next clock to be updated. The < operator for Clock
            # determines that the first clock to be updated should be the one
            # with the smallest t value, unless there are several with the 
            # same t value in which case it will be the one of those with the
            # smallest order value.
            clock = min(self._clocks)
        
    def stop(self):
        '''
        Stops the network from running, this is reset the next time ``run()`` is called.
        '''
        self._stopped = True
