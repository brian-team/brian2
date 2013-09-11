from brian2.core.base import BrianObject, Updater

__all__ = ['NetworkOperation', 'network_operation']


class NetworkOperation(BrianObject):
    """Object with function that is called every time step.
    
    Parameters
    ----------
    
    function : function
        The function to call every time step, should take either no arguments
        in which case it is called as ``function()`` or one argument, in which
        case it is called with the current `Clock` time (`Quantity`).
    when : `Scheduler`, optional
        Determines when the function should be called.
        
    See Also
    --------
    
    network_operation, Network, BrianObject
    """
    def __init__(self, function, when=None):
        BrianObject.__init__(self, when=when, name='networkoperation*')
        
        #: The function to be called each time step
        self.function = function
        
        if hasattr(function, 'func_code'):
            self._has_arg = (self.function.func_code.co_argcount==1)
        else:
            self._has_arg = False
            
        self.updater = NetworkOperationUpdater(self)            
        self.updaters[:] = [self.updater]
            
            
class NetworkOperationUpdater(Updater):
    def run(self):
        '''
        Call the `function`.
        '''
        op = self.owner
        if op._has_arg:
            op.function(op.clock.t)
        else:
            op.function()


def network_operation(*args, **kwds):
    """
    network_operation(when=None)
    
    Decorator to make a function get called every time step of a simulation.
    
    The function being decorated should either have no arguments, or a single
    argument which will be called with the current time `Clock.t`.
    
    Parameters
    ----------
    when : `Scheduler`, optional
        Determines when the function should be called.

    Examples
    --------
    
    Print something each time step:
    >>> from brian2 import *
    >>> @network_operation
    ... def f():
    ...   print('something')
    ...
    >>> net = Network(f)
    
    Print the time each time step:
    
    >>> @network_operation
    ... def f(t):
    ...   print('The time is', t)
    ...
    >>> net = Network(f)
    
    Specify a clock, etc.:
    
    >>> myclock = Clock(dt=0.5*ms) 
    >>> @network_operation(when=(myclock, 'start', 0))
    ... def f():
    ...   print('This will happen at the start of each timestep.')
    ...
    >>> net = Network(f)
    
    Notes
    -----
    
    Converts the function into a `NetworkOperation`.
    
    If using the form::
    
        @network_operations(when='start')
        def f():
            ...
            
    Then the arguments to network_operation must be keyword arguments.
    
    See Also
    --------
    
    NetworkOperation, Network, BrianObject
    """
    # Notes on this decorator:
    # Normally, a decorator comes in two types, with or without arguments. If
    # it has no arguments, e.g.
    #   @decorator
    #   def f():
    #      ...
    # then the decorator function is defined with an argument, and that
    # argument is the function f. In this case, the decorator function
    # returns a new function in place of f.
    #
    # However, you can also define:
    #   @decorator(arg)
    #   def f():
    #      ...
    # in which case the argument to the decorator function is arg, and the
    # decorator function returns a 'function factory', that is a callable
    # object that takes a function as argument and returns a new function.
    #
    # It might be clearer just to note that the first form above is equivalent
    # to:
    #   f = decorator(f)
    # and the second to:
    #   f = decorator(arg)(f)
    #
    # In this case, we're allowing the decorator to be called either with or
    # without an argument, so we have to look at the arguments and determine
    # if it's a function argument (in which case we do the first case above),
    # or if the arguments are arguments to the decorator, in which case we
    # do the second case above.
    #
    # Here, the 'function factory' is the locally defined class
    # do_network_operation, which is a callable object that takes a function
    # as argument and returns a NetworkOperation object.
    class do_network_operation(object):
        def __init__(self, **kwds):
            self.kwds = kwds
        def __call__(self, f):
            new_network_operation = NetworkOperation(f, **self.kwds)
            # Depending on whether we were called as @network_operation or
            # @network_operation(...) we need different levels, the level is
            # 2 in the first case and 1 in the second case (because in the
            # first case we go originalcaller->network_operation->do_network_operation
            # and in the second case we go originalcaller->do_network_operation
            # at the time when this method is called).
            new_network_operation.__name__ = f.__name__
            new_network_operation.__doc__ = f.__doc__
            new_network_operation.__dict__.update(f.__dict__)
            return new_network_operation
    if len(args)==1 and callable(args[0]):
        # We're in case (1), the user has written:
        # @network_operation
        # def f():
        #    ...
        # and the single argument to the decorator is the function f
        return do_network_operation()(args[0])
    else:
        # We're in case (2), the user has written:
        # @network_operation(...)
        # def f():
        #    ...
        # and the arguments must be keyword arguments
        return do_network_operation(**kwds)
