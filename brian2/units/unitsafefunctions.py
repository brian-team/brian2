"""
Unit-aware replacements for numpy functions.
"""
from functools import wraps

import pkg_resources
import numpy as np

from .fundamentalunits import (Quantity, wrap_function_dimensionless,
                               wrap_function_remove_dimensions,
                               fail_for_dimension_mismatch, is_dimensionless,
                               DIMENSIONLESS)

__all__ = [
         'log', 'log10', 'exp',
         'sin', 'cos', 'tan',
         'arcsin', 'arccos', 'arctan',
         'sinh', 'cosh', 'tanh',
         'arcsinh', 'arccosh', 'arctanh',
         'diagonal', 'ravel', 'trace', 'dot',
         'where',
         'ones_like', 'zeros_like',
         'arange', 'linspace'
         ]

def where(condition, *args, **kwds):  # pylint: disable=C0111
    if len(args) == 0:
        # nothing to do
        return np.where(condition, *args, **kwds)
    elif len(args) == 2:
        # check that x and y have the same dimensions
        fail_for_dimension_mismatch(args[0], args[1],
                                    'x and y need to have the same dimensions')

        if is_dimensionless(args[0]):
            return np.where(condition, *args, **kwds)
        else:
            # as both arguments have the same unit, just use the first one's
            dimensionless_args = [np.asarray(arg) for arg in args]
            return Quantity.with_dimensions(np.where(condition,
                                                     *dimensionless_args),
                                            args[0].dimensions)
    else:
        # illegal number of arguments, let numpy take care of this
        return np.where(condition, *args, **kwds)
where.__doc__ = np.where.__doc__

# Functions that work on dimensionless quantities only
sin = wrap_function_dimensionless(np.sin)
sinh = wrap_function_dimensionless(np.sinh)
arcsin = wrap_function_dimensionless(np.arcsin)
arcsinh = wrap_function_dimensionless(np.arcsinh)
cos = wrap_function_dimensionless(np.cos)
cosh = wrap_function_dimensionless(np.cosh)
arccos = wrap_function_dimensionless(np.arccos)
arccosh = wrap_function_dimensionless(np.arccosh)
tan = wrap_function_dimensionless(np.tan)
tanh = wrap_function_dimensionless(np.tanh)
arctan = wrap_function_dimensionless(np.arctan)
arctanh = wrap_function_dimensionless(np.arctanh)

log = wrap_function_dimensionless(np.log)
log10 = wrap_function_dimensionless(np.log10)
exp = wrap_function_dimensionless(np.exp)

ones_like = wrap_function_remove_dimensions(np.ones_like)
zeros_like = wrap_function_remove_dimensions(np.zeros_like)

def wrap_function_to_method(func):
    '''
    Wraps a function so that it calls the corresponding method on the
    Quantities object (if called with a Quantities object as the first
    argument). All other arguments are left untouched.
    '''
    @wraps(func)
    def f(x, *args, **kwds):  # pylint: disable=C0111
        if isinstance(x, Quantity):
            return getattr(x, func.__name__)(*args, **kwds)
        else:
            # no need to wrap anything
            return func(x, *args, **kwds)
    f.__doc__ = func.__doc__
    f.__name__ = func.__name__
    return f


@wraps(np.arange)
def arange(*args, **kwargs):
    # arange has a bit of a complicated argument structure unfortunately
    # we leave the actual checking of the number of arguments to numpy, though

    # default values
    start = kwargs.pop('start', 0)
    step = kwargs.pop('step', 1)
    stop = kwargs.pop('stop', None)
    if len(args) == 1:
        if stop is not None:
            raise TypeError('Duplicate definition of "stop"')
        stop = args[0]
    elif len(args) == 2:
        if start != 0:
            raise TypeError('Duplicate definition of "start"')
        if stop is not None:
            raise TypeError('Duplicate definition of "stop"')
        start, stop = args
    elif len(args) == 3:
        if start != 0:
            raise TypeError('Duplicate definition of "start"')
        if stop is not None:
            raise TypeError('Duplicate definition of "stop"')
        if step != 1:
            raise TypeError('Duplicate definition of "step"')
        start, stop, step = args
    elif len(args) > 3:
        raise TypeError('Need between 1 and 3 non-keyword arguments')
    if stop is None:
        raise TypeError('Missing stop argument.')
    fail_for_dimension_mismatch(start, stop,
                                error_message=('Start value {start} and stop '
                                               'value {stop} have to have the '
                                               'same units.'),
                                start=start, stop=stop)
    fail_for_dimension_mismatch(stop, step,
                                error_message=('Stop value {stop} and step '
                                               'value {step} have to have the '
                                               'same units.'),
                                stop=stop, step=step)
    dim = getattr(stop, 'dim', DIMENSIONLESS)
    return Quantity(np.arange(start=np.asarray(start),
                              stop=np.asarray(stop),
                              step=np.asarray(step),
                              **kwargs),
                    dim=dim, copy=False)

@wraps(np.linspace)
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None):
    fail_for_dimension_mismatch(start, stop,
                                error_message=('Start value {start} and stop '
                                               'value {stop} have to have the '
                                               'same units.'),
                                start=start, stop=stop)
    dim = getattr(start, 'dim', DIMENSIONLESS)
    if pkg_resources.parse_version(np.__version__) < pkg_resources.parse_version('1.9.0'):
        if dtype is not None:
            raise TypeError('The "dtype" argument needs numpy >= 1.9.0')
        result = np.linspace(np.asarray(start), np.asarray(stop), num=num,
                             endpoint=endpoint, retstep=retstep)
    else:
        result = np.linspace(np.asarray(start), np.asarray(stop), num=num,
                             endpoint=endpoint, retstep=retstep, dtype=dtype)
    return Quantity(result, dim=dim, copy=False)


# these functions discard subclass info -- maybe a bug in numpy?
ravel = wrap_function_to_method(np.ravel)
diagonal = wrap_function_to_method(np.diagonal)
trace = wrap_function_to_method(np.trace)
dot = wrap_function_to_method(np.dot)

# This is a very minor detail: setting the __module__ attribute allows the
# automatic reference doc generation mechanism to attribute the functions to
# this module. Maybe also helpful for IDEs and other code introspection tools.
sin.__module__ = __name__
sinh.__module__ = __name__
arcsin.__module__ = __name__
arcsinh.__module__ = __name__
cos.__module__ = __name__
cosh.__module__ = __name__
arccos.__module__ = __name__
arccosh.__module__ = __name__
tan.__module__ = __name__
tanh.__module__ = __name__
arctan.__module__ = __name__
arctanh.__module__ = __name__

log.__module__ = __name__
exp.__module__ = __name__
ravel.__module__ = __name__
diagonal.__module__ = __name__
trace.__module__ = __name__
dot.__module__ = __name__
arange.__module__ = __name__
linspace.__module__ = __name__

def setup():
    '''
    Setup function for doctests (used by nosetest).
    We do not want to test this module's docstrings as they
    are inherited from numpy.
    '''
    from nose import SkipTest
    raise SkipTest('Do not test numpy docstrings')
