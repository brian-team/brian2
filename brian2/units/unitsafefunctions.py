"""
Unit-aware replacements for numpy functions.
"""
from functools import wraps

import numpy as np

from .fundamentalunits import (Quantity, wrap_function_dimensionless,
                               fail_for_dimension_mismatch, is_dimensionless)

__all__ = [
         'log', 'exp',
         'sin', 'cos', 'tan',
         'arcsin', 'arccos', 'arctan',
         'sinh', 'cosh', 'tanh',
         'arcsinh', 'arccosh', 'arctanh',
         'diagonal', 'ravel', 'trace', 'dot',
         'where'
         ]

@wraps(np.where)
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
            return Quantity.with_dimensions(np.where(condition, *args),
                                            args[0].dimensions)
    else:
        # illegal number of arguments, let numpy take care of this
        return np.where(condition, *args, **kwds)

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
exp = wrap_function_dimensionless(np.exp)


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
    return f

# these functions discard subclass info -- maybe a bug in numpy?
ravel = wrap_function_to_method(np.ravel)
diagonal = wrap_function_to_method(np.diagonal)
trace = wrap_function_to_method(np.trace)
dot = wrap_function_to_method(np.dot)


def setup():
    '''
    Setup function for doctests (used by nosetest).
    We do not want to test this module's docstrings as they
    are inherited from numpy.
    '''
    from nose import SkipTest
    raise SkipTest('Do not test numpy docstrings')
