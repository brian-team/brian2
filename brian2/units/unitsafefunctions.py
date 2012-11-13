# ----------------------------------------------------------------------------------
# Copyright ENS, INRIA, CNRS
# Contributors: Romain Brette (brette@di.ens.fr) and Dan Goodman (goodman@di.ens.fr)
# 
# Brian is a computer program whose purpose is to simulate models
# of biological neural networks.
# 
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software.  You can  use, 
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info". 
# 
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability. 
# 
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or 
# data to be ensured and,  more generally, to use and operate it in the 
# same conditions as regards security. 
# 
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# ----------------------------------------------------------------------------------
# 
"""
Unit-aware replacements for numpy functions.
"""

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

def where(condition, *args):  # pylint: disable=C0111
    if len(args) == 0:
        # nothing to do
        return np.where(condition)
    elif len(args) == 2:
        # check that x and y have the same dimensions
        fail_for_dimension_mismatch(args[0], args[1],
                                    'x and y need to have the same dimensions')
        
        if is_dimensionless(args[0]):
            return np.where(condition, *args)
        else:
            # as both arguments have the same unit, just use the one from argument 1
            return Quantity.with_dimensions(np.where(condition, *args),
                                            args[0].dimensions)
    else:
        # illegal number of arguments, let numpy take care of this
        return np.where(condition, *args)
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
exp = wrap_function_dimensionless(np.exp)


def wrap_function_to_method(func):
    '''
    Wraps a function so that it calls the corresponding method on the Quantities
    object (if called with a Quantities object as the first argument). All
    other arguments are left untouched.
    '''
    def f(x, *args, **kwds):  # pylint: disable=C0111
        if isinstance(x, Quantity):
            return getattr(x, func.__name__)(*args, **kwds)
        else:
            # no need to wrap anything
            return func(x, *args, **kwds)
    f.__name__ = func.__name__
    f.__doc__ = func.__doc__
    if hasattr(func, '__dict__'):
        f.__dict__.update(func.__dict__)
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