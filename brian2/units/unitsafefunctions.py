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
Functions which check the dimensions of their arguments, etc.

Functions updated to provide Quantity functionality
---------------------------------------------------

With any dimensions:

* sqrt

Dimensionless:

* log, exp
* sin, cos, tan
* arcsin, arccos, arctan
* sinh, cosh, tanh
* arcsinh, arccosh, arctanh

With homogeneous dimensions:

* dot
"""
import numpy as np

from brian_unit_prefs import bup
from brian2.units.fundamentalunits import (Quantity, is_dimensionless,
                                           DimensionMismatchError, DIMENSIONLESS,
                                           wrap_function_dimensionless)


# these functions are the ones that will work with the template immediately below, and
# extend the numpy functions to know about Quantity objects 
__all__ = [
         'log', 'exp',
         'sin', 'cos', 'tan',
         'arcsin', 'arccos', 'arctan',
         'sinh', 'cosh', 'tanh',
         'arcsinh', 'arccosh', 'arctanh',
         'diagonal', 'ravel', 'trace'
         ]

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
    def f(x, *args, **kwds):
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
diagonal = wrap_function_to_method(np.ravel)
trace = wrap_function_to_method(np.ravel)