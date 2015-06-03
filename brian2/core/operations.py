import inspect

from brian2.core.base import BrianObject

__all__ = ['NetworkOperation', 'network_operation']


class NetworkOperation(BrianObject):
    """Object with function that is called every time step.
    
    Parameters
    ----------
    
    function : function
        The function to call every time step, should take either no arguments
        in which case it is called as ``function()`` or one argument, in which
        case it is called with the current `Clock` time (`Quantity`).
    dt : `Quantity`, optional
        The time step to be used for the simulation. Cannot be combined with
        the `clock` argument.
    clock : `Clock`, optional
        The update clock to be used. If neither a clock, nor the `dt` argument
        is specified, the `defaultclock` will be used.
    when : str, optional
        In which scheduling slot to execute the operation during a time step.
        Defaults to ``'start'``.
    order : int, optional
        The priority of this operation for operations occurring at the same time
        step and in the same scheduling slot. Defaults to 0.
        
    See Also
    --------
    
    network_operation, Network, BrianObject
    """
    add_to_magic_network = True
    def __init__(self, function, dt=None, clock=None, when='start', order=0):
        BrianObject.__init__(self, dt=dt, clock=clock, when=when, order=order, name='networkoperation*')

        #: The function to be called each time step
        self.function = function

        is_method = inspect.ismethod(function)

        if (hasattr(function, 'func_code') or  # Python 2
                hasattr(function, '__code__')):  # Python 3:
            argcount = function.func_code.co_argcount
            if is_method:
                if argcount == 2:
                    self._has_arg = True
                elif argcount == 1:
                    self._has_arg = False
                else:
                    raise TypeError(('Method "%s" cannot be used as a network '
                                     'operation, it needs to have either only '
                                     '"self" or "self, t" as arguments, but it '
                                     'has %d arguments.' % (function.__name__,
                                                            argcount)))
            else:
                if (argcount >= 1 and
                            function.func_code.co_varnames[0] == 'self'):
                    raise TypeError('The first argument of the function "%s" '
                                    'is "self", suggesting it is an instance '
                                    'method and not a function. Did you use '
                                    '@network_operation on a class method? '
                                    'This will not work, explicitly create a '
                                    'NetworkOperation object instead -- see '
                                    'the documentation for more '
                                    'details.' % function.__name__)
                if argcount == 1:
                    self._has_arg = True
                elif argcount == 0:
                    self._has_arg = False
                else:
                    raise TypeError(('Function "%s" cannot be used as a '
                                     'network operation, it needs to have '
                                     'either only "t" as an argument or have '
                                     'no arguments, but it has %d '
                                     'arguments.' % (function.__name__,
                                                     argcount)))
        else:
            self._has_arg = False

    def run(self):
        if self._has_arg:
            self.function(self._clock.t)
        else:
            self.function()


def network_operation(*args, **kwds):
    """
    network_operation(when=None)
    
    Decorator to make a function get called every time step of a simulation.
    
    The function being decorated should either have no arguments, or a single
    argument which will be called with the current time ``t``.
    
    Parameters
    ----------
    dt : `Quantity`, optional
        The time step to be used for the simulation. Cannot be combined with
        the `clock` argument.
    clock : `Clock`, optional
        The update clock to be used. If neither a clock, nor the `dt` argument
        is specified, the `defaultclock` will be used.
    when : str, optional
        In which scheduling slot to execute the operation during a time step.
        Defaults to ``'start'``.
    order : int, optional
        The priority of this operation for operations occurring at the same time
        step and in the same scheduling slot. Defaults to 0.

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
    
    Specify a dt, etc.:

    >>> @network_operation(dt=0.5*ms, when='end')
    ... def f():
    ...   print('This will happen at the end of each timestep.')
    ...
    >>> net = Network(f)
    
    Notes
    -----
    
    Converts the function into a `NetworkOperation`.
    
    If using the form::
    
        @network_operations(when='end')
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
