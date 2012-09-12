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
"""Defines physical units and quantities

The standard way to use this class is as follows:

V = 3 * volt
I = 2 * amp
R=V/I
print R

will return

1.5 ohm

The following fundamental units are defined:

metre, kilogram, second, amp, kelvin, mole, candle

And these additional basic units:

radian, steradian, hertz, newton, pascal, joule, watt,
coulomb, volt, farad, ohm, siemens, weber, tesla, henry,
celsius, lumen, lux, becquerel, gray, sievert, katal,
gram, gramme

Additionally, it includes all scaled versions of these
units using the standard SI prefixes (see the documentation
for the Unit class for more details), e.g. uamp,
mmetre, etc. It also includes the second and third powers
of each of these units, e.g. mvolt2 = mvolt*mvolt,
metre3 = metre**3, etc.

The module also defines these classes:

-- Dimension
        Stores the physical dimensions (length, mass, etc.)
-- DimensionMismatchError
        Exception raised if you try to add inconsistent units,
        etc.
-- Quantity
        The class of a value with a unit
-- Unit
        The class of the defined units like mvolt, etc.
-- UnitRegistry
        Stores 'known' units for printing

These functions:

-- get_dimensions(x)
        Returns the dimensions of a quantity or number x
-- have_same_dimensions(x,y)
        Tests if x and y have the same dimensions
-- is_dimensionless(x)
        Tests if x is dimensionless
-- display_in_unit(x,u)
        Displays quantity x in unit u
-- register_new_unit(u)
        Add a new unit u to the list of 'known' units for
        printing purposes
-- get_unit(x)
        Returns the fundamental unit of value x if one is known, or
        simply the value 1 with dimensions of x if none is known

And this decorator for function argument checking:

-- check_units(...)

If you want to use shorter named units, import the stdunits
module, which defines things like mV for mvolt, etc. They
are not included by default in the units module because of
the potential for variable name clashes.
"""
from warnings import warn
from operator import isNumberType, isSequenceType
from itertools import izip
from functools import wraps

import numpy as np

from brian_unit_prefs import bup

__all__ = [
    'DimensionMismatchError', 'get_or_create_dimension',
    'get_dimensions', 'is_dimensionless', 'have_same_dimensions',
    'display_in_unit', 'Quantity', 'Unit', 'register_new_unit',
    'check_units', 'is_scalar_type', 'get_unit', 'get_unit_fast',
    'scalar_representation',
    'unit_checking'
    ]

warn_if_no_unit_checking = True
unit_checking = True

#===============================================================================
# Numpy ufuncs
#===============================================================================

# Note: A list of numpy ufuncs can be found here:
# http://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs

# ufuncs that work on all dimensions and preserve the dimensions, e.g. abs
UFUNCS_PRESERVE_DIMENSIONS = ['absolute', 'rint', 'negative', 'conj',
                              'conjugate', 'floor', 'ceil', 'trunc']

# ufuncs that work on all dimensions but change the dimensions, e.g. square
UFUNCS_CHANGE_DIMENSIONS = ['multiply', 'divide', 'true_divide', 'floor_divide',
                            'sqrt', 'square', 'reciprocal', 'dot']

# ufuncs that work with matching dimensions, e.g. add
UFUNCS_MATCHING_DIMENSIONS = ['add', 'subtract', 'maximum', 'minimum',
                              'remainder', 'mod', 'fmod']

# ufuncs that compare values, i.e. work only with matching dimensions but do
# not result in a value with dimensions, e.g. equals
UFUNCS_COMPARISONS = ['less', 'less_equal', 'greater', 'greater_equal',
                      'equal', 'not_equal'] 

# Logical operations that work on all quantities and return boolean arrays
UFUNCS_LOGICAL = ['logical_and', 'logical_or', 'logical_xor', 'logical_not',
                  'isreal', 'iscomplex', 'isfinite', 'isinf', 'isnan']

# ufuncs that only work on dimensionless quantities
UFUNCS_DIMENSIONLESS = ['sin', 'sinh', 'arcsin', 'arcsinh', 'cos', 'cosh',
                        'arccos', 'arccosh', 'tan', 'tanh', 'arctan',
                        'arctanh', 'log', 'log2', 'log10', 'log1p',
                        'exp', 'exp2', 'expm1']

# ufuncs that only work on two dimensionless quantities
UFUNCS_DIMENSIONLESS_TWOARGS = ['logaddexp', 'logaddexp2', 'arctan2',
                                'hypot']

# ufuncs that only work on integers and therefore never on quantities
UFUNCS_INTEGERS = ['bitwise_and', 'bitwise_or', 'bitwise_xor', 'invert',
                   'left_shift', 'right_shift']


#==============================================================================
# Utility functions
#==============================================================================

def fail_for_dimension_mismatch(obj1, obj2=None, error_message=None):
    '''
    Raises a DimensionMismatchError if the dimensions of ``obj1`` and ``obj2``
    (can be scalars, arrays or Quantities) do not match. if ``obj2`` is None it
    is assumed to be dimensionless. An optional ``error_message`` can be given
    that is used for the DimensionMismatchError.
    
    Implements special checking for ``0``, treating it as having
    "any dimensions".
    '''
    if not unit_checking:
        return

    dim1 = get_dimensions(np.asanyarray(obj1))
    dim2 = get_dimensions(np.asanyarray(obj2)) if not obj2 is None else DIMENSIONLESS   

    if not dim1 is dim2:
        # Special treatment for "0":
        # if it is not a Quantity, it has "any dimension".
        # This allows expressions like 3 * mV + 0 to pass (useful in cases where
        # zero is treated as the neutral element, e.g. in the Python sum builtin)
        # or comparisons like 3 * mV == 0 to return False instead of failing
        # with a DimensionMismatchError. Note that 3 * mV == 0 * second or
        # 3 * mV == 0 * mV/mV is not allowed, though.
        if ((not isinstance(obj1, Quantity) and np.all(obj1 == 0)) or
            (not isinstance(obj2, Quantity) and np.all(obj2 == 0))):
            return

        if error_message is None:
            error_message = 'Dimension mismatch'
        raise DimensionMismatchError(error_message, dim1, dim2)


def wrap_function_dimensionless(func):
    '''
    Returns a new function that wraps the given function ``func``so that it
    raises a DimensionMismatchError if the function is called on a quantity with
    dimensions (excluding dimensionless quantitities). Quantities are
    transformed to unitless numpy arrays before calling ``func``.
    
    These checks/transformations apply only to the very first argument, all
    other arguments are ignored/untouched.
    '''
    def f(x, *args, **kwds):
        fail_for_dimension_mismatch(x, error_message=func.__name__)
        return func(np.asarray(x), *args, **kwds)
    f.__name__ = func.__name__
    f.__doc__ = func.__doc__
    if hasattr(func, '__dict__'):
        f.__dict__.update(func.__dict__)
    return f


def wrap_function_keep_dimensions(func):
    '''
    Returns a new function that wraps the given function ``func``so that it
    keeps the dimensions of its input. Quantities are transformed to
    unitless numpy arrays before calling ``func``, the output is a quantity
    with the original dimensions re-attached.
    
    These transformations apply only to the very first argument, all
    other arguments are ignored/untouched, allowing to work functions like
    ``sum`` to work as expected with additional ``axis`` etc. arguments.
    '''
    def f(x, *args, **kwds):
            return Quantity(func(np.asarray(x), *args, **kwds), dim=x.dim)
    f.__name__ = func.__name__
    f.__doc__ = func.__doc__
    if hasattr(func, '__dict__'):
        f.__dict__.update(func.__dict__)
    return f


def wrap_function_change_dimensions(func, change_dim_func):
    '''
    Returns a new function that wraps the given function ``func``so that it
    changes the dimensions of its input. Quantities are transformed to
    unitless numpy arrays before calling ``func``, the output is a quantity
    with the original dimensions passed through the function
    ``change_dim_func``. A typical use would be a ``sqrt`` function that uses
    ``lambda d: d ** 0.5`` as ``change_dim_func``.
    
    These transformations apply only to the very first argument, all
    other arguments are ignored/untouched.
    '''
    def f(x, *args, **kwds):
            ar = np.asarray(x)
            return Quantity(func(ar, *args, **kwds),
                            dim=change_dim_func(ar, x.dim))
    f.__name__ = func.__name__
    f.__doc__ = func.__doc__
    if hasattr(func, '__dict__'):
        f.__dict__.update(func.__dict__)
    return f


def wrap_function_remove_dimensions(func):
    '''
    Returns a new function that wraps the given function ``func``so that it
    removes any dimensions from its input. Useful for functions that are
    returning integers (indices) or booleans, irrespective of the datatype
    contained in the array.
    
    These transformations apply only to the very first argument, all
    other arguments are ignored/untouched.
    '''    
    def f(x, *args, **kwds):
            return func(np.asarray(x), *args, **kwds)
    f.__name__ = func.__name__
    f.__doc__ = func.__doc__
    if hasattr(func, '__dict__'):
        f.__dict__.update(func.__dict__)
    return f        


def wrap_function_no_check_warning(func):
    '''
    Returns a new function that wraps the given function ``func``so that it
    raises a warning about not checking units. Used mostly as a placeholder
    to make users aware that the function is not prepared for units yet.
    '''    
    def f(*args, **kwds):
            warn('%s does not check the units of its arguments.' % func.__name__)
            return func(*args, **kwds)
    f.__name__ = func.__name__
    f.__doc__ = func.__doc__
    if hasattr(func, '__dict__'):
        f.__dict__.update(func.__dict__)
    return f

# SI dimensions (see table at end of file) and various descriptions,
# each description maps to an index i, and the power of each dimension
# is stored in the variable dims[i]
_di = { "Length": 0, "length": 0, "metre": 0, "metres": 0, "meter": 0,
       "meters": 0, "m": 0,
       "Mass": 1, "mass": 1, "kilogram": 1, "kilograms": 1, "kg": 1,
       "Time": 2, "time": 2, "second": 2, "seconds": 2, "s": 2,
       "Electric Current":3, "electric current": 3, "Current": 3, "current": 3,
       "ampere": 3, "amperes": 3, "A": 3,
       "Temperature": 4, "temperature": 4, "kelvin": 4, "kelvins": 4, "K": 4,
       "Quantity of Substance": 5, "Quantity of substance": 5,
       "quantity of substance": 5, "Substance": 5, "substance": 5, "mole": 5,
       "moles": 5, "mol": 5,
       "Luminosity": 6, "luminosity": 6, "candle": 6, "candles": 6, "cd": 6 }

_ilabel = ["m", "kg", "s", "A", "K", "mol", "cd"]

# The same labels with the names used for constructing them in Python code
_iclass_label = ["metre", "kilogram", "second", "amp", "kelvin", "mole",
                 "candle"]

# SI unit _prefixes, see table at end of file
_siprefixes = {"y": 1e-24, "z": 1e-21, "a": 1e-18, "f": 1e-15, "p": 1e-12,
               "n": 1e-9, "u": 1e-6, "m": 1e-3, "c": 1e-2, "d": 1e-1, "": 1,
               "da": 1e1, "h": 1e2, "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12,
               "P": 1e15, "E": 1e18, "Z": 1e21, "Y": 1e24}


class _Dimension(object):
    '''Stores the indices of the 7 basic SI unit dimension (length, mass, etc.)
    
    Provides a subset of arithmetic operations appropriate to dimensions:
    multiplication, division and powers, and equality testing.
    
    Methods:
    
    is_dimensionless() returns Boolean value
    
    Notes:
    
    Users shouldn't use this class directly, but instead write things
    like:
    
    x = 3 * mvolt, etc.
    '''
    __slots__ = ["_dims"]
        
    __array_priority__ = 1000
    #### INITIALISATION ####
    
    def __init__(self, dims):
        '''
        Initializes a new :class:`_Dimension` object. This should never be
        done directly, use :func:`get_or_create_dimension` instead.
        '''
        self._dims = dims

    #### METHODS ####
    def get_dimension(self, d):
        #FIXME: incorrect docstring
        """Returns the list of dimension indices.
        
        See documentation for __init__.
        """
        return self._dims[_di[d]]

    def is_dimensionless(self):
        """Tells you whether the object is dimensionless."""
        return sum([x == 0 for x in self._dims]) == 7
    
    #### REPRESENTATION ####    
    def _str_representation(self, python_code=False):
        """String representation in basic SI units, or 1 for dimensionless.
        Use `python_code=False` for display purposes and `True` for valid
        Python code."""
        
        if python_code:
            power_operator = " ** "
        else:
            power_operator = "^"
        
        parts = []
        for i in range(len(self._dims)):
            if self._dims[i]:
                if python_code:
                    s = _iclass_label[i]
                else:
                    s = _ilabel[i]
                if self._dims[i] != 1:
                    s += power_operator + str(self._dims[i])
                parts.append(s)
        if python_code:
            s = " * ".join(parts)
            if not len(s):
                return "%s()" % self.__class__.__name__
        else:
            s = " ".join(parts)
            if not len(s):
                return "1"
        return s.strip()
    
    def __repr__(self):
        return self._str_representation(python_code=True)

    def __str__(self):
        return self._str_representation(python_code=False)
    
    #### ARITHMETIC ####
    # Note that none of the dimension arithmetic objects do sanity checking
    # on their inputs, although most will throw an exception if you pass the
    # wrong sort of input
    def __mul__(self, value):
        return get_or_create_dimension([x + y for x, y in izip(self._dims, value._dims)])

    def __div__(self, value):
        return get_or_create_dimension([x - y for x, y in izip(self._dims, value._dims)])

    def __truediv__(self, value):
        return self.__div__(value)

    def __pow__(self, value):
        value = np.asarray(value)
        if value.size> 1:
            raise ValueError('Too many exponents')
        return get_or_create_dimension([x * value for x in self._dims])

    def __imul__(self, value):
        raise TypeError('Dimension object is immutable')

    def __idiv__(self, value):
        raise TypeError('Dimension object is immutable')

    def __itruediv__(self, value):
        raise TypeError('Dimension object is immutable')

    def __ipow__(self, value):
        raise TypeError('Dimension object is immutable')

    #### COMPARISON ####
    def __eq__(self, value):
        return np.allclose(self._dims, value._dims)

    def __ne__(self, value):
        return not self.__eq__(value)

    #### MAKE DIMENSION PICKABLE ####
    def __getstate__(self):
        return self._dims

    def __setstate__(self, state):
        self._dims = state

DIMENSIONLESS = _Dimension((0, 0, 0, 0, 0, 0, 0))
_dimensions = {(0, 0, 0, 0, 0, 0, 0): DIMENSIONLESS}


def get_or_create_dimension(*args, **kwds):
    """Get a _Dimension object with a vector or keywords. This takes care of
    only creating new objects if they were not created before and otherwise
    returning a reference to an existing object. This allows to compare
    dimensions very efficiently using ``is``.
    
    Call as get_or_create_dimension(list/tuple) or
    get_or_create_dimension(keywords)
    
    list/tuple -- a list or tuple with the indices of the 7 elements of an SI dimension
    keywords -- a sequence of keyword=value pairs where the keywords are
      the names of the SI dimensions, or the standard unit
    
    Examples:
    
    The following are all definitions of the dimensions of force
    
    get_or_create_dimension(length=1, mass=1, time=-2)
    get_or_create_dimension(m=1, kg=1, s=-2)
    get_or_create_dimension([1,1,-2,0,0,0,0])
    
    The 7 units are (in order):
    
    Length, Mass, Time, Electric Current, Temperature,
    Quantity of Substance, Luminosity
    
    and can be referred to either by these names or their SI unit names,
    e.g. length, metre, and m all refer to the same thing here.
    """
    if len(args):
        if isSequenceType(args[0]) and len(args[0]) == 7:
            # initialisation by list
            dims = args[0]
        else:
            raise ValueError('Need a sequence of exactly 7 items')
    else:
        # initialisation by keywords
        dims = [0, 0, 0, 0, 0, 0, 0]
        for k in kwds.keys():
            # _di stores the index of the dimension with name 'k'
            dims[_di[k]] = kwds[k]

    dims = tuple(dims)
    
    # check whether this _Dimension object has already been created
    if dims in _dimensions:
        return _dimensions[dims]
    else:
        new_dim = _Dimension(dims)
        _dimensions[dims] = new_dim
        return new_dim


class DimensionMismatchError(Exception):
    """Exception class for attempted operations with inconsistent dimensions
    
    For example, ``3*mvolt + 2*amp`` raises this exception. The purpose of this
    class is to help catch errors based on incorrect units. The exception will
    print a representation of the dimensions of the two inconsistent objects
    that were operated on. If you want to check for inconsistent units in your
    code, do something like::
    
        try:
            ...
            your code here
            ...
        except DimensionMismatchError, inst:
            ...
            cleanup code here, e.g.
            print "Found dimension mismatch, details:", inst
            ...
    """
    def __init__(self, description, *dims):
        """Raise as DimensionMismatchError(desc,dim1,dim2,...)
        
        desc -- a description of the type of operation being performed, e.g.
                Addition, Multiplication, etc.
        dim -- the dimensions of the objects involved in the operation, any
               number of them is possible
        """
        # Call the base class constructor to make Exception pickable, see:
        # http://bugs.python.org/issue1692335
        Exception.__init__(self, description, *dims)
        self.dims = dims
        self.desc = description

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = self.desc + ", dimensions were "
        for d in self.dims:
            s += "(" + str(d) + ") "
        return s

def is_scalar_type(obj):
    """Tells you if the object is a 1d number type
    
    This function is mostly used internally by the module for
    argument type checking. A scalar type can be considered
    a dimensionless quantity (see the documentation for
    Quantity for more information).
    """
    return isNumberType(obj) and not isSequenceType(obj)


def get_dimensions(obj):
    """Returns the dimensions of any object that has them.
    
    Slightly more general than obj.get_dimensions() because it will return
    a new dimensionless Dimension() object if the object is of number type
    but not a Quantity (e.g. a float or int).
    """
    if isNumberType(obj) and not isinstance(obj, Quantity):
        return DIMENSIONLESS
    return obj.get_dimensions()


def is_dimensionless(obj):
    """Tests if a scalar value is dimensionless or not, returns a ``bool``.
    
    Note that the syntax may change in later releases of Brian, with tighter
    integration of scalar and array valued quantities.
    """
    return get_dimensions(obj) is DIMENSIONLESS


def have_same_dimensions(obj1, obj2):
    """Tests if two scalar values have the same dimensions, returns a ``bool``.
    
    Note that the syntax may change in later releases of Brian, with tighter
    integration of scalar and array valued quantities.
    """
    # If dimensions are consistently created using get_or_create_dimensions,
    # the fast "is" comparison should always return the correct result.
    # To be safe, we also do an equals comparison in case it fails. This
    # should only add a small amount of unnecessary computation for cases in
    # which this function returns False which very likely leads to a
    # DimensionMismatchError.
    return (get_dimensions(obj1) is get_dimensions(obj2) or
            get_dimensions(obj1) == get_dimensions(obj2))


def display_in_unit(x, u):
    """String representation of the object x in unit u.
    """
    fail_for_dimension_mismatch(x, u,
                                "Non-matching unit for function display_in_unit")

    s = str(np.asarray(x / u)) + " "
    if not is_dimensionless(u):
        if isinstance(u, Unit):
            s += str(u)
        else:
            s += str(u.dim)
    return s.strip()

def quantity_with_dimensions(floatval, dims):
    '''
    Create a new :class:`Quantity` object with the given units. Calls
    :func:`get_or_create_dimensions` with the ``dims`` argument to make sure
    that unpickling (which calls this function) does not accidentally create
    new :class:`_Dimension` objects which should instead refer to existing
    ones.
    '''
    return Quantity.with_dimensions(floatval,
                                    get_or_create_dimension(dims._dims))


class Quantity(np.ndarray):
    """A number with an associated physical dimension.
    
    In most cases, it is not necessary to create a :class:`Quantity` object
    by hand, instead use the constant unit names ``second``, ``kilogram``,
    etc. 

    This is the main user class for the units module, although
    in most cases it is not necessary to initialise a new
    quantity by hand (see construction below for details).
    
    The Quantity class defines arithmetic operations which
    check for consistency of dimensions and raise the
    DimensionMismatchError exception if they are inconsistent.
    
    The class also defines default and other representations
    of a number for printing purposes.
    
    Typical usage:
    
    I = 3 * amp # I is a Quantity object
    R = 2 * ohm # same for R
    print I * R # displays "6 V"
    print (I * R).in_unit(mvolt) # displays "6000 mV"
    print (I * R) / mvolt # displays "6000"
    x = I + R # raises DimensionMismatchError
    
    Is = np.array([1, 2, 3]) * amp #Is is a Quantity object
    print Is * R # displays "[ 2.  4.  6.] V"
    print np.asarray(Is * R) # displays "[2. 4. 6.]" (no units)
    
    See the documentation on the Unit class for more details
    about the available unit names like mvolt, etc.
    
    Casting rules:
    
    The rules that define the casting operations for
    Quantity object are:
    
    1. Quantity op Quantity = Quantity
        - Performs dimension checking if appropriate
    2. (Scalar or Array) op Quantity = Quantity 
        - Assumes that the scalar or array is dimensionless

    Construction details:
          
    x = Quantity.with_dimensions(value,dim) returns an object with
        floating point value value, and dimensions dim, see the
        documentation for Quantity.with_dimensions(...) for more.

    Static constructors:
    
    -- with_dimensions(dim)
    -- with_dimensions(keywords...)
    
    Methods:
    
    -- get_dimensions() return Dimension
    -- set_dimensions(dim)
    -- is_dimensionless() return boolean
    -- at_scale(scale) return string
    -- has_same_dimensions(other) return boolean
    -- in_unit(unit) return string
    -- in_best_unit() return string
    """
    __slots__ = ["dim"]
    
    __array_priority__ = 1000
    
    #===========================================================================
    # Construction and handling of numpy ufuncs
    #===========================================================================
    def __new__(cls, arr, dim=None, dtype=None, copy=False):
        # All np.ndarray subclasses need something like this, see
        # http://www.scipy.org/Subclasses
        subarr = np.array(arr, dtype=dtype, copy=copy).view(cls)

        # Use the given dimension or the dimension of the given array (if any)
        if hasattr(arr, 'dim'):
            subarr.dim = arr.dim
            if not (dim is None) and not (dim is subarr.dim):
                raise DimensionMismatchError('Conflicting dimension '
                                             'information between array and '
                                             'dim keyword',
                                             arr.dim, dim)
        elif not isinstance(arr, np.ndarray):
            # check whether it is an iterable containing Quantity objects
            try:
                is_quantity = [isinstance(x, Quantity) for x in arr]
                if all(is_quantity):
                    dims = [x.dim for x in arr]
                    one_dim = dims[0]
                    for d in dims:
                        if d != one_dim:
                            raise DimensionMismatchError('Mixing quantities '
                                                         'with different '
                                                         'dimensions is not '
                                                         'allowed',
                                                         d, one_dim)
                    subarr.dim = dims[0]
                    if not (dim is None) and not (dim is subarr.dim):
                        raise DimensionMismatchError('Conflicting dimension '
                                                     'information between '
                                                     'sequence and dim keyword',
                                                     subarr.dim, dim)
                elif any(is_quantity):
                    raise ValueError('Mixing quantities and non-quantities is '
                                     'not allowed.')

            except TypeError:
                # Not iterable
                pass

        if dim is not None:
            subarr.dim = dim
        elif not hasattr(subarr, 'dim'):
            subarr.dim = DIMENSIONLESS

        return subarr

    def __array_finalize__(self, orig):
        # If we already have a dimension, check that it is consistent
        if hasattr(self, 'dim'):
            if hasattr(orig, 'dim') and not (self.dim is orig.dim):
                # TODO: Better error message
                raise DimensionMismatchError('Mismatching dimensions', self.dim,
                                             orig.dim)
        else:
            self.dim = getattr(orig, 'dim', DIMENSIONLESS)

    def __array_prepare__(self, array, context=None):
        if context is None:
            return array

        uf, args, _ = context

        if uf.__name__ in (UFUNCS_PRESERVE_DIMENSIONS +
                           UFUNCS_CHANGE_DIMENSIONS + 
                           UFUNCS_LOGICAL):
            # always allowed
            pass
        elif uf.__name__ in UFUNCS_INTEGERS:
            # Numpy should already raise a TypeError by itself
            raise TypeError('%s cannot be used on quantities.' % uf.__name__)
        elif uf.__name__ in UFUNCS_MATCHING_DIMENSIONS + UFUNCS_COMPARISONS:
            # Ok if dimension of arguments match 
            fail_for_dimension_mismatch(args[0], args[1], uf.__name__)        
        elif uf.__name__ in UFUNCS_DIMENSIONLESS:
            # Ok if argument is dimensionless
            fail_for_dimension_mismatch(args[0], error_message=uf.__name__)
        elif uf.__name__ in UFUNCS_DIMENSIONLESS_TWOARGS:
            # Ok if both arguments are dimensionless
            fail_for_dimension_mismatch(args[0], error_message=uf.__name__)
            fail_for_dimension_mismatch(args[1], error_message=uf.__name__)
        elif uf.__name__ == 'power':            
            fail_for_dimension_mismatch(args[1], error_message=uf.__name__)
            if np.asarray(args[1]).size != 1:
                raise TypeError('Only length-1 arrays can be used as an '
                                'exponent for quantities.')
        elif uf.__name__ in ('sign', 'ones_like'):
            return np.asarray(array)
        else:
            warn("Unknown ufunc '%s' in __array_prepare__" % uf.__name__)            

        return array

    def __array_wrap__(self, array, context=None):
        dim = DIMENSIONLESS

        if not context is None:
            uf, args, _ = context
            if uf.__name__ in UFUNCS_PRESERVE_DIMENSIONS + UFUNCS_MATCHING_DIMENSIONS:
                dim = self.dim
            elif uf.__name__ in UFUNCS_DIMENSIONLESS + UFUNCS_DIMENSIONLESS_TWOARGS:
                # We should have been arrived here only for dimensionless
                # quantities
                dim = DIMENSIONLESS
            elif uf.__name__ in (UFUNCS_COMPARISONS +
                                 UFUNCS_LOGICAL + 
                                 ['sign', 'ones_like']):
                # Do not touch the return value (boolean or integer array)
                return array
            elif uf.__name__ == 'sqrt':
                dim = self.dim ** 0.5
            elif uf.__name__ == 'power':
                dim = get_dimensions(args[0]) ** np.asarray(args[1])
            elif uf.__name__ == 'square':
                dim = self.dim ** 2
            elif uf.__name__ in ('divide', 'true_divide', 'floor_divide'):
                dim = get_dimensions(args[0]) / get_dimensions(args[1])
            elif uf.__name__ == 'reciprocal':
                dim = get_dimensions(1 / get_dimensions(args[0]))
            elif uf.__name__ in ('multiply', 'dot'):
                dim = get_dimensions(args[0]) * get_dimensions(args[1])
            else:
                warn("Unknown ufunc '%s' in __array_wrap__" % uf.__name__)
                #TODO: Remove units in this case?

        # This seems to be better than using type(self) instead of quantity
        # This may convert units to Quantities, e.g. np.square(volt) leads to
        # a 1 * volt ** 2 quantitiy instead of volt ** 2. But this should
        # rarely be an issue. The alternative leads to more confusing
        # behaviour: np.float64(3) * mV would result in a dimensionless float64         
        result = array.view(Quantity)
        result.dim = dim
        return result

#===============================================================================
# Quantity-specific functions (not existing in ndarray)
#===============================================================================
    @staticmethod
    def with_dimensions(value, *args, **keywords):
        """Static method to create a Quantity object with dimensions
        
        Use as Quantity.with_dimensions(value,dim),
               Quantity.with_dimensions(value,dimlist) or
               Quantity.with_dimensions(value,keywords...)
               
        -- value is a float or other scalar type
        -- dim is a dimension object
        -- dimlist, keywords (see the Dimension constructor)
        
        e.g.
        
        x = Quantity.with_dimensions(2,Dimension(length=1))
        x = Quantity.with_dimensions(2,length=1)
        x = 2 * metre
        
        all define the same object.
        """
        x = Quantity(value)
        if len(args) and isinstance(args[0], _Dimension):
            x.set_dimensions(args[0])
        else:
            x.set_dimensions(get_or_create_dimension(*args, **keywords))
        return x

    #### METHODS ####
    def get_dimensions(self):
        """Returns the dimensions of this object
        """
        return self.dim

    def set_dimensions(self, dim):
        """Set the dimensions of this object
        """
        self.dim = dim

    def is_dimensionless(self):
        """Tells you whether this is a dimensionless object
        """
        return self.dim.is_dimensionless()

    def at_scale(self, scale):
        # FIXME, do not use Scale object, move some functions to _Dimension
        """Returns a string representation at given scale
        """
        return (str(np.asarray(self) / scale.scale_factor(self.dim)) + " " +
                scale.unit_representation(self.dim))

    def has_same_dimensions(self, other):
        """Tells you if this object has the same dimensions as another.
        """
        return self.dim == get_dimensions(other)

    def in_unit(self, u, python_code=False):
        """String representation of the object in unit `u`.
        If `python_code` is `True`, this will return valid python code, i.e. a
        string like `5.0 * um ** 2`instead of `5.0 um^2`  
        """
        
        fail_for_dimension_mismatch(self, u,
                                    "Non-matching unit for method in_unit")
        if python_code:
            s = repr(np.asarray(self / u)) + " "
        else:
            s = str(np.asarray(self / u)) + " "
        if not u.is_dimensionless():
            if isinstance(u, Unit):
                if python_code:                    
                    s += '* ' + repr(u)
                else:
                    s += str(u)
            else:
                if python_code:
                    s += "* " + repr(u.dim)
                else:
                    s += str(u.dim)
        elif python_code == True:  # A quantity without unit is not recognisable otherwise
            return '%s(%s)' % (self.__class__.__name__, s.strip())
        return s.strip()

    def in_best_unit(self, python_code=False, *regs):
        """String representation of the object in the 'best unit'
        
        If `python_code` is `True`, this will return valid python code, i.e. a
        string like `5.0 * um ** 2`instead of `5.0 um^2`  
        
        For more information, see the documentation for the UnitRegistry
        class. Essentially, this looks at the value of the quantity for
        all 'known' matching units (e.g. mvolt, namp, etc.) and returns
        the one with the most compact representation. Standard units are
        built in, but you can register new units for consideration. 
        """
        u = _get_best_unit(self, *regs)
        return self.in_unit(u, python_code)

#===============================================================================
# Overwritten ndarray methods
#===============================================================================

    #### Setting/getting items ####
    def __getitem__(self, key):
        ''' Overwritten to assure that single elements (i.e., indexed with a
        single integer) retain their unit.
        '''
        if np.isscalar(key) and np.issubdtype(type(key), np.integer):
            return Quantity.with_dimensions(super(Quantity, self).__getitem__(key),
                                            self.dim)
        else:
            return super(Quantity, self).__getitem__(key)

    def __getslice__(self, start, end):
        return self.__getitem__(slice(start, end))

    def __setitem__(self, key, value):
        fail_for_dimension_mismatch(self, value,
                                    'Inconsistent units in assignment')
        return super(Quantity, self).__setitem__(key, value)

    def __setslice__(self, start, end, value):
        return self.__setitem__(slice(start, end), value)

    #### ARITHMETIC ####
    def __mul__(self, other):
        if isinstance(other, np.ndarray) or is_scalar_type(other):
            return Quantity.with_dimensions(np.asarray(self) * np.asarray(other),
                                            self.dim * get_dimensions(other))
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        if isinstance(other, np.ndarray) or is_scalar_type(other):
            super(Quantity, self).__imul__(other)
            self.dim = self.dim * get_dimensions(other)
            return self
        else:
            return NotImplemented

    def __div__(self, other):
        if isinstance(other, np.ndarray) or is_scalar_type(other):
            return Quantity.with_dimensions(np.asarray(self) / np.asarray(other),
                                            self.dim / get_dimensions(other))
        else:
            return NotImplemented

    def __truediv__(self, other):
        return self.__div__(other)

    def __rdiv__(self, other):
        if isinstance(other, np.ndarray) or is_scalar_type(other):
            return Quantity.with_dimensions(np.asarray(other) / np.asarray(self),
                                            get_dimensions(other) / self.dim)
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        return self.__rdiv__(self, other)

    def __idiv__(self, other):
        if isinstance(other, np.ndarray) or is_scalar_type(other):
            super(Quantity, self).__idiv__(other)
            self.dim = self.dim / get_dimensions(other)
            return self
        else:
            return NotImplemented

    def __itruediv__(self, other):
        if isinstance(other, np.ndarray) or is_scalar_type(other):
            super(Quantity, self).__itruediv__(other)
            self.dim = self.dim / get_dimensions(other)
            return self
        else:
            return NotImplemented

    def __mod__(self, other):
        if isinstance(other, np.ndarray) or is_scalar_type(other):
            fail_for_dimension_mismatch(self, other, 'Modulo')
            return Quantity.with_dimensions(np.asarray(self) % np.asarray(other),
                                            self.dim)
        else:
            return NotImplemented

    def __add__(self, other):
        if isinstance(other, np.ndarray) or is_scalar_type(other):
            fail_for_dimension_mismatch(self, other, 'Addition')
            return Quantity.with_dimensions(np.asarray(self) + np.asarray(other),
                                            self.dim)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, np.ndarray) or is_scalar_type(other):
            fail_for_dimension_mismatch(self, other, 'Addition')
            super(Quantity, self).__iadd__(other)
            return self
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, np.ndarray) or is_scalar_type(other):
            fail_for_dimension_mismatch(self, other, 'Subtraction')
            return Quantity.with_dimensions(np.asarray(self) - np.asarray(other),
                                            self.dim)
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, np.ndarray) or is_scalar_type(other):
            fail_for_dimension_mismatch(self, other, 'Subtraction (R)')
            return Quantity.with_dimensions(np.asarray(other) - np.asarray(self),
                                                self.dim)
        else:
            return NotImplemented

    def __isub__(self, other):
        if isinstance(other, np.ndarray) or is_scalar_type(other):
            fail_for_dimension_mismatch(self, other, 'Subtraction')
            super(Quantity, self).__isub__(other)
            return self
        else:
            return NotImplemented

    def __pow__(self, other):
        if isinstance(other, np.ndarray) or is_scalar_type(other):
            fail_for_dimension_mismatch(other, error_message='Power')
            # FIXME do not allow multiple values for exponent
            return Quantity.with_dimensions(np.asarray(self) ** np.asarray(other),
                                            self.dim ** np.asarray(other))
        else:
            return NotImplemented

    def __rpow__(self, other):
        if self.is_dimensionless():
            if isinstance(other, np.ndarray) or isinstance(other, np.ndarray):
                return Quantity.with_dimensions(np.asarray(other) ** np.asarray(self),
                                                DIMENSIONLESS)
            else:
                return NotImplemented
        else:
            raise DimensionMismatchError("Power(R)", self.dim)

    def __ipow__(self, other):
        if isinstance(other, np.ndarray) or is_scalar_type(other):
            fail_for_dimension_mismatch(other, error_message='Power')
            # FIXME do not allow multiple values for exponent
            super(Quantity, self).__ipow__(np.asarray(other))
            self.dim = self.dim ** np.asarray(other)
            return self
        else:
            return NotImplemented

    def __neg__(self):
        return Quantity.with_dimensions(-np.asarray(self), self.dim)

    def __pos__(self):
        return self

    def __abs__(self):
        return Quantity.with_dimensions(abs(np.asarray(self)), self.dim)

    def tolist(self):
        def replace_with_quantity(seq, dim):
            def top_replace(s):
                for i in s:
                    if not isinstance(i, list):
                        yield Quantity.with_dimensions(i, dim)
                    else:
                        yield type(i)(top_replace(i))

            return type(seq)(top_replace(seq))
        return replace_with_quantity(np.asarray(self).tolist(), self.dim)

    #### COMPARISONS ####
    def __lt__(self, other):
        is_scalar = is_scalar_type(other)
                
        if is_scalar:
            # special handling of Inf and -Inf
            if np.isposinf(other):
                return True
            if np.isneginf(other):
                return False
        
        fail_for_dimension_mismatch(self, other, 'LessThan')
    
        if isinstance(other, np.ndarray) or is_scalar:    
            return np.asarray(self) < np.asarray(other)
        else:
            return NotImplemented

    def __le__(self, other):
        is_scalar = is_scalar_type(other)
                
        if is_scalar:
            # special handling of Inf and -Inf
            if np.isposinf(other):
                return True
            if np.isneginf(other):
                return False
        
        fail_for_dimension_mismatch(self, other, 'LessThanOrEquals')
    
        if isinstance(other, np.ndarray) or is_scalar:    
            return np.asarray(self) <= np.asarray(other)
        else:
            return NotImplemented


    def __gt__(self, other):
        is_scalar = is_scalar_type(other)
                
        if is_scalar:
            # special handling of Inf and -Inf
            if np.isposinf(other):
                return True
            if np.isneginf(other):
                return False
        
        fail_for_dimension_mismatch(self, other, 'GreaterThan')
    
        if isinstance(other, np.ndarray) or is_scalar:    
            return np.asarray(self) > np.asarray(other)
        else:
            return NotImplemented

    def __ge__(self, other):
        is_scalar = is_scalar_type(other)
                
        if is_scalar:
            # special handling of Inf and -Inf
            if np.isposinf(other):
                return True
            if np.isneginf(other):
                return False
        
        fail_for_dimension_mismatch(self, other, 'GreaterThanOrEquals')
    
        if isinstance(other, np.ndarray) or is_scalar:    
            return np.asarray(self) >= np.asarray(other)
        else:
            return NotImplemented

    def __eq__(self, other):
        is_scalar = is_scalar_type(other)
                
        if is_scalar:
            # special handling of Inf and -Inf
            if np.isposinf(other):
                return True
            if np.isneginf(other):
                return False
        
        fail_for_dimension_mismatch(self, other, 'Equals')
    
        if isinstance(other, np.ndarray) or is_scalar:    
            return np.asarray(self) == np.asarray(other)
        else:
            return NotImplemented

    def __ne__(self, other):
        is_scalar = is_scalar_type(other)
                
        if is_scalar:
            # special handling of Inf and -Inf
            if np.isposinf(other):
                return True
            if np.isneginf(other):
                return False
        
        fail_for_dimension_mismatch(self, other, 'NotEquals')
    
        if isinstance(other, np.ndarray) or is_scalar:    
            return np.asarray(self) != np.asarray(other)
        else:
            return NotImplemented

    #### MAKE QUANTITY PICKABLE ####
    def __reduce__(self):
        return (quantity_with_dimensions, (np.asarray(self), self.dim))


    #### REPRESENTATION ####
    def __repr__(self):
        return self.in_best_unit(python_code=True)

    def __str__(self):
        return self.in_best_unit()

    #### Mathematic methods ####

    cumsum = wrap_function_keep_dimensions(np.ndarray.cumsum)
    diagonal = wrap_function_keep_dimensions(np.ndarray.diagonal)
    max = wrap_function_keep_dimensions(np.ndarray.max)
    mean = wrap_function_keep_dimensions(np.ndarray.mean)
    min = wrap_function_keep_dimensions(np.ndarray.min)
    ptp = wrap_function_keep_dimensions(np.ndarray.ptp)
    ravel = wrap_function_keep_dimensions(np.ndarray.ravel)
    round = wrap_function_keep_dimensions(np.ndarray.round)
    std = wrap_function_keep_dimensions(np.ndarray.std)
    sum = wrap_function_keep_dimensions(np.ndarray.sum)
    trace = wrap_function_keep_dimensions(np.ndarray.trace)
    var = wrap_function_change_dimensions(np.ndarray.var, lambda ar, d: d ** 2)
    all = wrap_function_remove_dimensions(np.ndarray.all)
    any = wrap_function_remove_dimensions(np.ndarray.any)
    argmax = wrap_function_remove_dimensions(np.ndarray.argmax)
    argmin = wrap_function_remove_dimensions(np.ndarray.argmax)
    argsort = wrap_function_remove_dimensions(np.ndarray.argsort)
    
    def fill(self, values):
        fail_for_dimension_mismatch(self, values, 'fill')
        super(Quantity, self).fill(values)
    fill.__doc__ = np.ndarray.fill.__doc__

    def put(self, indices, values, *args, **kwds):
        fail_for_dimension_mismatch(self, values, 'fill')
        super(Quantity, self).put(indices, values, *args, **kwds)
    put.__doc__ = np.ndarray.put.__doc__

    def clip(self, a_min, a_max, *args, **kwds):
        fail_for_dimension_mismatch(self, a_min, 'clip')
        fail_for_dimension_mismatch(self, a_max, 'clip')        
        return super(Quantity, self).clip(np.asarray(a_min),
                                          np.asarray(a_max),
                                          *args, **kwds)
    clip.__doc__ = np.ndarray.clip.__doc__

    def dot(self, other, **kwds):
        return Quantity.with_dimensions(np.array(self).dot(np.array(other)),
                                        self.dim * get_dimensions(other))
    dot.__doc__ = np.ndarray.dot.__doc__

    def searchsorted(self, v, **kwds):
        fail_for_dimension_mismatch(self, v, 'searchsorted')
        return super(Quantity, self).searchsorted(np.asarray(v))
    searchsorted.__doc__ = np.ndarray.searchsorted.__doc__

    def prod(self, *args, **kwds):
        prod_result = super(Quantity, self).prod(*args, **kwds)
        # Calculating the correct dimensions is not completly trivial (e.g.
        # like doing self.dim**self.size) because prod can be called on
        # multidimensional arrays along a certain axis. 
        # Our solution: Use a "dummy matrix" containing a 1 (without units) at
        # each entry and sum it, using the same keyword arguments as provided.
        # The result gives the exponent for the dimensions.
        # This relies on sum and prod having the same arguments, which is true
        # now and probably remains like this in the future 
        dim_exponent = np.ones_like(self).sum(*args, **kwds)
        # The result is possibly multidimensional but all entries should be
        # identical
        if dim_exponent.size > 1:
            dim_exponent = dim_exponent[0]
        return Quantity.with_dimensions(prod_result, self.dim ** dim_exponent)
    prod.__doc__ = np.ndarray.prod.__doc__        

    def cumprod(self, *args, **kwds):
        if not self.is_dimensionless():
            raise ValueError('cumprod over array elements on quantities '
                             'with dimensions is not possible.')
        return Quantity(np.asarray(self).cumprod(*args, **kwds))
    cumprod.__doc__ = np.ndarray.cumprod.__doc__


class Unit(Quantity):
    '''
    A physical unit
    
    Normally, you do not need to worry about the implementation of
    units. They are derived from the :class:`Quantity` object with
    some additional information (name and string representation).
    You can define new units which will be used when generating
    string representations of quantities simply by doing an
    arithmetical operation with only units, for example::
    
        Nm = newton * metre
    
    Note that operations with units are slower than operations with
    :class:`Quantity` objects, so for efficiency if you do not need the
    extra information that a :class:`Unit` object carries around, write
    ``1*second`` in preference to ``second``.
    '''

    # original documentation
    """A physical unit
    
    Basically, a unit is just a quantity with given dimensions, e.g.
    mvolt = 0.001 with the dimensions of voltage. The units module
    defines a large number of standard units, and you can also define
    your own (see below).
    
    The unit class also keeps track of various things that were used
    to define it so as to generate a nice string representation of it.
    See Representation below.
    
    Typical usage:
    
    x = 3 * mvolt # returns a quantity
    print x.in_unit(uvolt) # returns 3000 uV 
    
    Standard units:
    
    The units class has the following fundamental units:
    
    metre, kilogram, second, amp, kelvin, mole, candle
    
    And these additional basic units:
    
    radian, steradian, hertz, newton, pascal, joule, watt,
    coulomb, volt, farad, ohm, siemens, weber, tesla, henry,
    celsius, lumen, lux, becquerel, gray, sievert, katal
    
    And additionally, it includes all scaled versions of these
    units using the following prefixes

     Factor     Name    Prefix
     -----      ----    ------
     10^24      yotta   Y
     10^21      zetta   Z
     10^18      exa     E
     10^15      peta    P
     10^12      tera    T
     10^9       giga    G
     10^6       mega    M
     10^3       kilo    k
     10^2       hecto   h
     10^1       deka    da
     1           
     10^-1      deci    d
     10^-2      centi   c
     10^-3      milli   m
     10^-6      micro   u (\mu in SI)
     10^-9      nano    n
     10^-12     pico    p
     10^-15     femto   f
     10^-18     atto    a
     10^-21     zepto   z
     10^-24     yocto   y
    
    So for example nohm, ytesla, etc. are all defined.
    
    Defining your own:
    
    It can be useful to define your own units for printing
    purposes. So for example, to define the newton metre, you
    write:
    
    Nm = newton * metre
    
    Writing:
    
    print (1*Nm).in_unit(Nm)
    
    will return "1 Nm" because the Unit class generates a new
    display name of "Nm" from the display names "N" and "m" for
    newtons and metres automatically (see Representation below).
    
    To register this unit for use in the automatic printing
    of the Quantity.in_best_unit() method, see the documentation
    for the UnitRegistry class.
    
    Construction:
    
    The best way to construct a new unit is to use standard units
    already defined and arithmetic operations, e.g. newton*metre.
    See the documentation for __init__ and the static methods create(...)
    and create_scaled_units(...) for more details.
    
    If you don't like the automatically generated display name for
    the unit, use the set_display_name(name) method.
    
    Representation:
    
    A new unit defined by multiplication, division or taking powers
    generates a name for the unit automatically, so that for
    example the name for pfarad/mmetre**2 is "pF/mm^2", etc. If you
    don't like the automatically generated name, use the 
    set_display_name(name) method.
    """
    __slots__ = ["dim", "scale", "scalefactor", "dispname", "name",
                 "iscompound"]
    
    __array_priority__ = 100
    
    #### CONSTRUCTION ####
    def __new__(cls, arr, dim=None, scale=None, dtype=None, copy=False):
        obj = super(Unit, cls).__new__(cls, arr, dim=dim, dtype=dtype,
                                       copy=copy)
        global automatically_register_units
        if automatically_register_units:
            register_new_unit(obj)
        return obj
    
    def __array_finalize__(self, orig):
        self.dim = getattr(orig, 'dim', DIMENSIONLESS)
        self.scale = getattr(orig, 'scale', ("", "", "", "", "", "", ""))
        self.scalefactor = getattr(orig, 'scalefactor', '')
        self.name = getattr(orig, 'name', '')
        self.dispname = getattr(orig, 'dispname', '')
        self.iscompound = getattr(orig, 'iscompound', False)
        return self  
        
    def __init__(self, value, dim=None, scale=None):
        """Initialises a unit
        """
        super(Unit, self).__init__(value)
        self.dim = dim
        if scale is None:
            scale = ("", "", "", "", "", "", "")
        if not len(scale) == 7:
            raise ValueError('scale needs seven entries')
        self.scale = scale
        self.scalefactor = ""
        self.name = ""
        self.dispname = ""
        self.iscompound = False

    @staticmethod
    def create(dim, name="", dispname="", scalefactor="", **keywords):
        """Creates a new named unit
        
        dim -- the dimensions of the unit
        name -- the full name of the unit, e.g. volt
        dispname -- the display name, e.g. V
        scalefactor -- scaling factor, e.g. m for mvolt
        keywords -- the scaling for each SI dimension, e.g. length="m", mass="-1", etc.
        """
        scale = [ "", "", "", "", "", "", "" ]
        for k in keywords:
            scale[_di[k]] = keywords[k]
        v = 1.0
        for s, i in izip(scale, dim._dims):
            if i: v *= _siprefixes[s] ** i
        u = Unit(v * _siprefixes[scalefactor], dim=dim, scale=tuple(scale))
        u.scalefactor = scalefactor + ""
        u.name = name + ""
        u.dispname = dispname + ""
        u.iscompound = False
        return u

    @staticmethod
    def create_scaled_unit(baseunit, scalefactor):
        """Create a scaled unit from a base unit
        
        baseunit -- e.g. volt, amp
        scalefactor -- e.g. "m" for mvolt, mamp
        """
        u = Unit(np.asarray(baseunit) * _siprefixes[scalefactor],
                 dim=baseunit.dim, scale=baseunit.scale)
        u.scalefactor = scalefactor
        u.name = scalefactor + baseunit.name
        u.dispname = scalefactor + baseunit.dispname
        u.iscompound = False
        return u

    #### METHODS ####
    def set_name(self, name):
        """Sets the name for the unit
        """
        self.name = name

    def set_display_name(self, name):
        """Sets the display name for the unit
        """
        self.dispname = name

    #### REPRESENTATION ####  
    def __repr__(self):
        if self.name == "":
            if self.scalefactor:
                parts = [repr(_siprefixes[self.scalefactor])] 
            else:
                parts = []
            for i in range(7):
                if self.dim._dims[i]:                
                    s = self.scale[i] + _iclass_label[i]
                    if self.dim._dims[i] != 1:
                        s += ' ** ' + str(self.dim._dims[i])
                    parts.append(s)
            s = " * ".join(parts) 
            if not len(s):
                return "%s(1)" % self.__class__.__name__
            return s.strip()
        else:
            return self.name

    def __str__(self):
        if self.dispname == "":
            s = self.scalefactor + " "
            for i in range(7):
                if self.dim._dims[i]:
                    s += self.scale[i] + _ilabel[i]
                    if self.dim._dims[i] != 1:
                        s += "^" + str(self.dim._dims[i])
                    s += " "
            s = s.strip()
            if not len(s):
                return "1"
            return s
        else:
            return self.dispname

    #### ARITHMETIC ####
    def __mul__(self, other):
        if isinstance(other, Unit):
            u = Unit(np.asarray(self) * np.asarray(other))
            u.name = self.name + " * " + other.name
            u.dispname = self.dispname + ' ' + other.dispname
            u.dim = self.dim * other.dim
            u.iscompound = True
            return u
        else:
            return super(Unit, self).__mul__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        if isinstance(other, Unit):
            u = Unit(np.asarray(self) / np.asarray(other))            
            if other.iscompound:
                u.dispname = '(' + self.dispname + ')'
                u.name = '(' + self.name + ')'
            else:
                u.dispname = self.dispname
                u.name = self.name
            u.dispname += '/'
            u.name += ' / '
            if other.iscompound:
                u.dispname += '(' + other.dispname + ')'
                u.name += '(' + other.name + ')'
            else:
                u.dispname += other.dispname
                u.name += other.name
            u.dim = self.dim / other.dim
            u.iscompound = True
            return u
        else:
            return super(Unit, self).__div__(other)

    def __rdiv__(self, other):
        if isinstance(other, Unit):
            return other.__div__(self)
        else:
            return super(Unit, self).__rdiv__(other)

    def __pow__(self, other):
        if is_scalar_type(other):
            u = Unit(np.asarray(self) ** other)            
            if self.iscompound:
                u.dispname = '(' + self.dispname + ')'
                u.name = '(' + self.name + ')'
            else:
                u.dispname = self.dispname
                u.name = self.name
            u.dispname += '^' + str(other)
            u.name += ' ** ' + repr(other)
            u.dim = self.dim ** other
            return u
        else:
            return super(Unit, self).__pow__(other)

    def __iadd__(self, other):
        raise TypeError('Units cannot be modified in-place')
    
    def __isub__(self, other):
        raise TypeError('Units cannot be modified in-place')
    
    def __imul__(self, other):
        raise TypeError('Units cannot be modified in-place')
    
    def __idiv__(self, other):
        raise TypeError('Units cannot be modified in-place')

    def __itruediv__(self, other):
        raise TypeError('Units cannot be modified in-place')
    
    def __ifloordiv__(self, other):
        raise TypeError('Units cannot be modified in-place')
        
    def __imod__(self, other):
        raise TypeError('Units cannot be modified in-place')
    
    def __ipow__(self, other, modulo=None):
        raise TypeError('Units cannot be modified in-place')

automatically_register_units = True

class UnitRegistry(object):
    """Stores known units for printing in best units
    
    All a user needs to do is to use the register_new_unit(u)
    function.
    
    Default registries:
    
    The units module defines three registries, the standard units,
    user units, and additional units. Finding best units is done
    by first checking standard, then user, then additional. New
    user units are added by using the register_new_unit(u) function.
    
    Standard units includes all the basic non-compound unit names
    built in to the module, including volt, amp, etc. Additional
    units defines some compound units like newton metre (Nm) etc.
    
    Methods:
    
    add(u) - add a new unit
    __getitem__(x) - get the best unit for quantity x
      e.g. UnitRegistry ur; ur[3*mvolt] returns mvolt
    """
    def __init__(self):
        self.objs = []

    def add(self, u):
        """Add a unit to the registry
        """
        self.objs.append(u)

    def __getitem__(self, x):
        """Returns the best unit for quantity x
        
        The algorithm is to consider the value:
        
        m=abs(x/u)
        
        for all matching units u. If there is a unit u with a value of
        m in [1,1000) then we select that unit. Otherwise, we select
        the first matching unit (which will typically be the unscaled
        version).
        """
        matching = filter(lambda o: have_same_dimensions(o, x), self.objs)
        if len(matching) == 0:
            raise KeyError("Unit not found in registry.")
        # count the number of entries well represented by this unit 
        floatreps = np.asarray(map(lambda o:
                                    np.sum(np.logical_and(0.1 <= abs(np.asarray(x / o)),
                                                                abs(np.asarray(x / o)) < 100)),
                                    matching))
        if any(floatreps):
            return matching[floatreps.argmax()]
        else:
            return matching[0]

def register_new_unit(u):
    """Register a new unit for automatic displaying of quantities
    
    Example usage:
    
    2.0*farad/metre**2 = 2.0 m^-4 kg^-1 s^4 A^2
    register_new_unit(pfarad / mmetre**2)
    2.0*farad/metre**2 = 2000000.0 pF/mm^2
    """
    UserUnitRegister.add(u)

standard_unit_register = UnitRegistry()
additional_unit_register = UnitRegistry()
UserUnitRegister = UnitRegistry()

def all_registered_units(*regs):
    """Returns all registered units in the correct order
    """
    if not len(regs):
        regs = [ standard_unit_register, UserUnitRegister, additional_unit_register]
    for r in regs:
        for u in r.objs:
            yield u

def _get_best_unit(x, *regs):
    """Returns the best unit for quantity x
    
    Checks the registries regs, unless none are provided in which
    case it will check the standard, user and additional unit
    registers in turn.
    """
    if get_dimensions(x) is DIMENSIONLESS:
        return Quantity(1)
    if len(regs):
        for r in regs:
            try:
                return r[x]
            except KeyError:
                pass
        return Quantity.with_dimensions(1, x.dim)
    else:
        return _get_best_unit(x, standard_unit_register, UserUnitRegister,
                              additional_unit_register)

def get_unit(x, *regs):
    '''
    Find the most appropriate consistent unit from the unit registries, or just
    return a Quantity with the same dimensions and value 1.
    '''
    for u in all_registered_units(*regs):
        if np.asarray(u) == 1 and have_same_dimensions(u, x):
            return u
    return Quantity.with_dimensions(1, get_dimensions(x))

def get_unit_fast(x):
    '''
    Return a quantity with value 1 and the same dimensions.
    '''
    return Quantity.with_dimensions(1, get_dimensions(x))

#### DECORATORS


def check_units(**au):
    """Decorator to check units of arguments passed to a function
    
    **Sample usage:** ::
    
        @check_units(I=amp,R=ohm,wibble=metre,result=volt)
        def getvoltage(I,R,**k):
            return I*R

    You don't have to check the units of every variable in the function, and
    you can define what the units should be for variables that aren't
    explicitly named in the definition of the function. For example, the code
    above checks that the variable wibble should be a length, so writing::
    
        getvoltage(1*amp,1*ohm,wibble=1)
    
    would fail, but::
    
        getvoltage(1*amp,1*ohm,wibble=1*metre)
    
    would pass.
    String arguments are not checked (e.g. ``getvoltage(wibble='hello')`` would pass).
    
    The special name ``result`` is for the return value of the function.
    
    An error in the input value raises a :exc:`DimensionMismatchError`, and an error
    in the return value raises an ``AssertionError`` (because it is a code
    problem rather than a value problem).
    
    **Notes**
    
    This decorator will destroy the signature of the original function, and
    replace it with the signature ``(*args, **kwds)``. Other decorators will
    do the same thing, and this decorator critically needs to know the signature
    of the function it is acting on, so it is important that it is the first
    decorator to act on a function. It cannot be used in combination with another
    decorator that also needs to know the signature of the function.
    """
    def do_check_units(f):
        @wraps(f)
        def new_f(*args, **kwds):
            newkeyset = kwds.copy()
            arg_names = f.func_code.co_varnames[0:f.func_code.co_argcount]
            for (n, v) in zip(arg_names, args[0:f.func_code.co_argcount]):
                newkeyset[n] = v
            for k in newkeyset.iterkeys():
                # string variables are allowed to pass, the presumption is they
                # name another variable. None is also allowed, useful for 
                # default parameters
                if (k in au.keys() and not isinstance(newkeyset[k], str) and
                                       not newkeyset[k] is None): 
                    if not have_same_dimensions(newkeyset[k], au[k]):
                        raise DimensionMismatchError("Function " + f.__name__ +
                                                     " variable " + k +
                                                     " should have dimensions of " +
                                                     str(au[k]),
                                                     get_dimensions(newkeyset[k]))
            result = f(*args, **kwds)
            if "result" in au:
                assert have_same_dimensions(result, au["result"]), \
                    ("Function " + f.__name__ + " should return a value with unit " +
                     str(au["result"]) + " but has returned " +
                     str(get_dimensions(result)))
            return result
        return new_f
    return do_check_units


def _check_nounits(**au):
    """Don't bother checking units decorator
    """
    def dont_check_units(f):
        return f
    return dont_check_units

# TODO: What is this used for?
def scalar_representation(x):
    if isinstance(x, Unit):
        return x.name
    u = get_unit(x)
    if isinstance(u, Unit):
        return '(' + repr(np.asarray(x)) + '*' + u.name + ')'
    if isinstance(x, Quantity):
        return ('(Quantity.with_dimensions(' + repr(np.asarray(x)) + ',' +
                repr(x.dim._dims) + '))')
    return repr(x)

# Remove all units
if not bup.use_units:
    check_units = _check_nounits
    def get_dimensions(obj):
        return DIMENSIONLESS

    def is_dimensionless(obj):
        return True

    def have_same_dimensions(obj1, obj2):
        return True

    def get_unit(x, *regs):
        return 1.

    def scalar_representation(x):
        return '1.0'

###################################################
##### ADDITIONAL INFORMATION

#SI DIMENSIONS
#-------------
#Quantity               Unit      Symbol
#--------               ----      ------
#Length                 metre     m
#Mass                   kilogram  kg
#Time                   second    s
#Electric current       ampere    A
#Temperature            kelvin    K
#Quantity of substance  mole      mol
#Luminosity             candle    cd

# SI UNIT PREFIXES
# ----------------
# Factor     Name    Prefix
# -----      ----    ------
# 10^24      yotta   Y
# 10^21      zetta   Z
# 10^18      exa     E
# 10^15      peta    P
# 10^12      tera    T
# 10^9       giga    G
# 10^6       mega    M
# 10^3       kilo    k
# 10^2       hecto   h
# 10^1       deka    da
# 1           
# 10^-1      deci    d
# 10^-2      centi   c
# 10^-3      milli   m
# 10^-6      micro   u (\mu in SI)
# 10^-9      nano    n
# 10^-12     pico    p
# 10^-15     femto   f
# 10^-18     atto    a
# 10^-21     zepto   z
# 10^-24     yocto   y