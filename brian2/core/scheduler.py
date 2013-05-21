from brian2.core.clocks import Clock, defaultclock

__all__ = ['Scheduler']

class Scheduler(object):
    '''
    Scheduler(clock=None, when='start', order=0)
    
    Defines which `BrianObject` gets called in what order in `Network.run`.
    
    Any time a `Scheduler` object is required in an argument, you can instead
    give a `Clock`, string (for the ``when`` argument), int or float (for
    the ``order`` argument), or a tuple of any combination of these, e.g.
    you can write ``(clock, 'start', 1)`` instead of
    ``Scheduler(clock=clock, when='start', order=1)``, or ``('end', 2)`` 
    instead of ``Scheduler(when='end', order=2)``.
        
    Parameters
    ----------
    clock : `Clock`, optional
        The update clock determining when the `BrianObject` will be updated, or
        use `defaultclock` if unspecified.
    when : str, optional
        Defines when the `BrianObject` is updated in the main `Network.run`
        loop.
    order : (int, float), optional
        A `BrianObject` with the same ``when`` value will be updated in order
        of increasing values of ``order``, or if both are equal then the order
        is unspecified (but will always be the same on each iteration).

    Notes
    -----
    The `Scheduler` object is just a convenience class, and is not stored by
    `BrianObject`, only its attributes.
    '''
    def __init__(self, *args, **kwds):
        if len(args) and len(kwds):
            raise TypeError("Scheduler has to be specified by a sequence of "
                            "arguments or a set of keyword arguments, but not "
                            "both.")
        if len(args)==1 and isinstance(args[0], Scheduler):
            arg = args[0]
            clock = arg.clock
            order = arg.order
            when = arg.when
            defined_clock = arg.defined_clock
            defined_order = arg.defined_order
            defined_when = arg.defined_when
        elif kwds:
            defined_clock = 'clock' in kwds
            defined_when = 'when' in kwds
            defined_order = 'order' in kwds
            clock = kwds.pop('clock', defaultclock)
            when = kwds.pop('when', 'start')
            order = kwds.pop('order', 0)
            if kwds:
                raise TypeError("Scheduler does not take keyword arguments "
                                ""+", ".join(kwds.keys()))
        else:
            defined_clock = False
            defined_when = False
            defined_order = False
            clock = None
            when = None
            order = None
            if len(args)>1 or args[0] is not None:
                if len(args)==1 and not isinstance(args[0], (Clock, str, int, float)):
                    args = args[0]
                for arg in args:
                    if isinstance(arg, Clock):
                        if clock is not None:
                            raise TypeError("Should only specify one clock.")
                        clock = arg
                        defined_clock = True
                    elif isinstance(arg, str):
                        if when is not None:
                            raise TypeError("Should only specify one when.")
                        when = arg
                        defined_when = True
                    elif isinstance(arg, (int, float)):
                        if order is not None:
                            raise TypeError("Should only specify one order.")
                        order = arg
                        defined_order = True
                    else:
                        raise TypeError("Arguments should be CLock, string, int or float")
            if clock is None: clock = defaultclock
            if when is None: when = 'start'
            if order is None: order = 0
        
        #: The `Clock` determining when the object should be updated.
        self.clock = clock
        
        #: The ID string determining when the object should be updated in `Network.run`.   
        self.when = when
        
        #: The order in which objects with the same clock and ``when`` should be updated
        self.order = order

        #: Whether or not the user explicitly specified a clock
        self.defined_clock = defined_clock
        
        #: Whether or not the user explicitly specified a when
        self.defined_when = defined_when
        
        #: Whether or not the user explicitly specified an order
        self.defined_order = defined_order

    def __repr__(self):
        description = '{classname}(clock={clock}, when={when}, order={order})'
        return description.format(classname=self.__class__.__name__,
                                  clock=repr(self.clock),
                                  when=repr(self.when),
                                  order=repr(self.order))
