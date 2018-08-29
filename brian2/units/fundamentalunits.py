# coding=utf-8
"""
Defines physical units and quantities

=====================  ========  ======
Quantity               Unit      Symbol
---------------------  --------  ------
Length                 metre     m
Mass                   kilogram  kg
Time                   second    s
Electric current       ampere    A
Temperature            kelvin    K
Quantity of substance  mole      mol
Luminosity             candle    cd
=====================  ========  ======
"""
from __future__ import division

import numbers
import collections
from warnings import warn
import operator
import itertools

import numpy as np
from numpy import VisibleDeprecationWarning
from sympy import latex

__all__ = [
    'DimensionMismatchError', 'get_or_create_dimension',
    'get_dimensions', 'is_dimensionless', 'have_same_dimensions',
    'in_unit', 'in_best_unit', 'Quantity', 'Unit', 'register_new_unit',
    'check_units', 'is_scalar_type', 'get_unit',
    ]


unit_checking = True

def _flatten(iterable):
    '''
    Flatten a given list `iterable`.
    '''
    for e in iterable:
        if isinstance(e, list):
            for f in _flatten(e):
                yield f
        else:
            yield e


def _short_str(arr):
    '''
    Return a short string representation of an array, suitable for use in
    error messages.
    '''
    arr = np.asanyarray(arr)
    old_printoptions = np.get_printoptions()
    np.set_printoptions(edgeitems=2, threshold=5)
    arr_string = str(arr)
    np.set_printoptions(**old_printoptions)
    return arr_string

#===============================================================================
# Numpy ufuncs
#===============================================================================

# Note: A list of numpy ufuncs can be found here:
# http://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs

#: ufuncs that work on all dimensions and preserve the dimensions, e.g. abs
UFUNCS_PRESERVE_DIMENSIONS = ['absolute', 'rint', 'negative', 'conj',
                              'conjugate', 'floor', 'ceil', 'trunc']

#: ufuncs that work on all dimensions but change the dimensions, e.g. square
UFUNCS_CHANGE_DIMENSIONS = ['multiply', 'divide', 'true_divide',
                            'floor_divide', 'sqrt', 'square', 'reciprocal',
                            'dot']

#: ufuncs that work with matching dimensions, e.g. add
UFUNCS_MATCHING_DIMENSIONS = ['add', 'subtract', 'maximum', 'minimum',
                              'remainder', 'mod', 'fmod']

#: ufuncs that compare values, i.e. work only with matching dimensions but do
#: not result in a value with dimensions, e.g. equals
UFUNCS_COMPARISONS = ['less', 'less_equal', 'greater', 'greater_equal',
                      'equal', 'not_equal']

#: Logical operations that work on all quantities and return boolean arrays
UFUNCS_LOGICAL = ['logical_and', 'logical_or', 'logical_xor', 'logical_not',
                  'isreal', 'iscomplex', 'isfinite', 'isinf', 'isnan']

#: ufuncs that only work on dimensionless quantities
UFUNCS_DIMENSIONLESS = ['sin', 'sinh', 'arcsin', 'arcsinh', 'cos', 'cosh',
                        'arccos', 'arccosh', 'tan', 'tanh', 'arctan',
                        'arctanh', 'log', 'log2', 'log10', 'log1p',
                        'exp', 'exp2', 'expm1']

#: ufuncs that only work on two dimensionless quantities
UFUNCS_DIMENSIONLESS_TWOARGS = ['logaddexp', 'logaddexp2', 'arctan2',
                                'hypot']

#: ufuncs that only work on integers and therefore never on quantities
UFUNCS_INTEGERS = ['bitwise_and', 'bitwise_or', 'bitwise_xor', 'invert',
                   'left_shift', 'right_shift']


#==============================================================================
# Utility functions
#==============================================================================

def fail_for_dimension_mismatch(obj1, obj2=None, error_message=None,
                                **error_quantities):
    '''
    Compare the dimensions of two objects.

    Parameters
    ----------
    obj1, obj2 : {array-like, `Quantity`}
        The object to compare. If `obj2` is ``None``, assume it to be
        dimensionless
    error_message : str, optional
        An error message that is used in the DimensionMismatchError
    error_quantities : dict mapping str to `Quantity`, optional
        Quantities in this dictionary will be converted using the `_short_str`
        helper method and inserted into the ``error_message`` (which should
        have placeholders with the corresponding names). The reason for doing
        this in a somewhat complicated way instead of directly including all the
        details in ``error_messsage`` is that converting large quantity arrays
        to strings can be rather costly and we don't want to do it if no error
        occured.

    Returns
    -------
    dim1, dim2 : `Dimension`, `Dimension`
        The physical dimensions of the two arguments (so that later code does
        not need to get the dimensions again).

    Raises
    ------
    DimensionMismatchError
        If the dimensions of `obj1` and `obj2` do not match (or, if `obj2` is
        ``None``, in case `obj1` is not dimensionsless).

    Notes
    -----
    Implements special checking for ``0``, treating it as having "any
    dimensions".
    '''
    if not unit_checking:
        return None, None

    dim1 = get_dimensions(obj1)
    if obj2 is None:
        dim2 = DIMENSIONLESS
    else:
        dim2 = get_dimensions(obj2)

    if dim1 is not dim2 and not (dim1 is None or dim2 is None):
        # Special treatment for "0":
        # if it is not a Quantity, it has "any dimension".
        # This allows expressions like 3*mV + 0 to pass (useful in cases where
        # zero is treated as the neutral element, e.g. in the Python sum
        # builtin) or comparisons like 3 * mV == 0 to return False instead of
        # failing # with a DimensionMismatchError. Note that 3*mV == 0*second
        # is not allowed, though.
        if ((dim1 is DIMENSIONLESS and np.all(obj1 == 0)) or
                (dim2 is DIMENSIONLESS and np.all(obj2 == 0))):
            return dim1, dim2

        # We do another check here, this should allow Brian1 units to pass as
        # having the same dimensions as a Brian2 unit
        if dim1 == dim2:
            return dim1, dim2

        if error_message is None:
            error_message = 'Dimension mismatch'
        else:
            error_quantities = {name: _short_str(q)
                                for name, q in error_quantities.iteritems()}
            error_message = error_message.format(**error_quantities)
        # If we are comparing an object to a specific unit, we don't want to
        # restate this unit (it is probably mentioned in the text already)
        if obj2 is None or isinstance(obj2, (Dimension, Unit)):
            raise DimensionMismatchError(error_message, dim1)
        else:
            raise DimensionMismatchError(error_message, dim1, dim2)
    else:
        return dim1, dim2


def wrap_function_dimensionless(func):
    '''
    Returns a new function that wraps the given function `func` so that it
    raises a DimensionMismatchError if the function is called on a quantity
    with dimensions (excluding dimensionless quantities). Quantities are
    transformed to unitless numpy arrays before calling `func`.

    These checks/transformations apply only to the very first argument, all
    other arguments are ignored/untouched.
    '''
    def f(x, *args, **kwds): # pylint: disable=C0111
        fail_for_dimension_mismatch(x,
                                    error_message=('%s expects a dimensionless '
                                                   'argument but got '
                                                   '{value}' % func.__name__),
                                    value=x)
        return func(np.array(x, copy=False), *args, **kwds)
    f._arg_units = [1]
    f._return_unit = 1
    f.__name__ = func.__name__
    f.__doc__ = func.__doc__
    return f


def wrap_function_keep_dimensions(func):
    '''
    Returns a new function that wraps the given function `func` so that it
    keeps the dimensions of its input. Quantities are transformed to
    unitless numpy arrays before calling `func`, the output is a quantity
    with the original dimensions re-attached.

    These transformations apply only to the very first argument, all
    other arguments are ignored/untouched, allowing to work functions like
    ``sum`` to work as expected with additional ``axis`` etc. arguments.
    '''
    def f(x, *args, **kwds):  # pylint: disable=C0111
        return Quantity(func(np.array(x, copy=False), *args, **kwds), dim=x.dim)
    f._arg_units = [None]
    f._return_unit = lambda u : u
    f.__name__ = func.__name__
    f.__doc__ = func.__doc__
    return f


def wrap_function_change_dimensions(func, change_dim_func):
    '''
    Returns a new function that wraps the given function `func` so that it
    changes the dimensions of its input. Quantities are transformed to
    unitless numpy arrays before calling `func`, the output is a quantity
    with the original dimensions passed through the function
    `change_dim_func`. A typical use would be a ``sqrt`` function that uses
    ``lambda d: d ** 0.5`` as ``change_dim_func``.

    These transformations apply only to the very first argument, all
    other arguments are ignored/untouched.
    '''
    def f(x, *args, **kwds):  # pylint: disable=C0111
        ar = np.array(x, copy=False)
        return Quantity(func(ar, *args, **kwds),
                        dim=change_dim_func(ar, x.dim))
    f._arg_units = [None]
    f._return_unit = change_dim_func
    f.__name__ = func.__name__
    f.__doc__ = func.__doc__
    return f


def wrap_function_remove_dimensions(func):
    '''
    Returns a new function that wraps the given function `func` so that it
    removes any dimensions from its input. Useful for functions that are
    returning integers (indices) or booleans, irrespective of the datatype
    contained in the array.

    These transformations apply only to the very first argument, all
    other arguments are ignored/untouched.
    '''
    def f(x, *args, **kwds):  # pylint: disable=C0111
        return func(np.array(x, copy=False), *args, **kwds)
    f._arg_units = [None]
    f._return_unit = 1
    f.__name__ = func.__name__
    f.__doc__ = func.__doc__
    return f


# SI dimensions (see table at the top of the file) and various descriptions,
# each description maps to an index i, and the power of each dimension
# is stored in the variable dims[i]
_di = {"Length": 0, "length": 0, "metre": 0, "metres": 0, "meter": 0,
       "meters": 0, "m": 0,
       "Mass": 1, "mass": 1, "kilogram": 1, "kilograms": 1, "kg": 1,
       "Time": 2, "time": 2, "second": 2, "seconds": 2, "s": 2,
       "Electric Current":3, "electric current": 3, "Current": 3, "current": 3,
       "ampere": 3, "amperes": 3, "A": 3,
       "Temperature": 4, "temperature": 4, "kelvin": 4, "kelvins": 4, "K": 4,
       "Quantity of Substance": 5, "Quantity of substance": 5,
       "quantity of substance": 5, "Substance": 5, "substance": 5, "mole": 5,
       "moles": 5, "mol": 5,
       "Luminosity": 6, "luminosity": 6, "candle": 6, "candles": 6, "cd": 6}

_ilabel = ["m", "kg", "s", "A", "K", "mol", "cd"]

# The same labels with the names used for constructing them in Python code
_iclass_label = ["metre", "kilogram", "second", "amp", "kelvin", "mole",
                 "candle"]

# SI unit _prefixes as integer exponents of 10, see table at end of file.
_siprefixes = {"y": -24, "z": -21, "a": -18, "f": -15, "p": -12,
               "n": -9, "u": -6, "m": -3, "c": -2, "d": -1, "": 0,
               "da": 1, "h": 2, "k": 3, "M": 6, "G": 9, "T": 12,
               "P": 15, "E": 18, "Z": 21, "Y": 24}


class Dimension(object):
    '''
    Stores the indices of the 7 basic SI unit dimension (length, mass, etc.).

    Provides a subset of arithmetic operations appropriate to dimensions:
    multiplication, division and powers, and equality testing.

    Parameters
    ----------
    dims : sequence of `float`
        The dimension indices of the 7 basic SI unit dimensions.

    Notes
    -----
    Users shouldn't use this class directly, it is used internally in Quantity
    and Unit. Even internally, never use ``Dimension(...)`` to create a new
    instance, use `get_or_create_dimension` instead. This function makes
    sure that only one Dimension instance exists for every combination of
    indices, allowing for a very fast dimensionality check with ``is``.
    '''
    __slots__ = ["_dims"]

    __array_priority__ = 1000
    #### INITIALISATION ####

    def __init__(self, dims):
        self._dims = dims

    #### METHODS ####
    def get_dimension(self, d):
        """
        Return a specific dimension.

        Parameters
        ----------
        d : `str`
            A string identifying the SI basic unit dimension. Can be either a
            description like "length" or a basic unit like "m" or "metre".

        Returns
        -------
        dim : `float`
            The dimensionality of the dimension `d`.
        """
        return self._dims[_di[d]]

    @property
    def is_dimensionless(self):
        '''
        Whether this Dimension is dimensionless.

        Notes
        -----
        Normally, instead one should check dimension for being identical to
        `DIMENSIONLESS`.
        '''
        return all([x == 0 for x in self._dims])

    @property
    def dim(self):
        '''
        Returns the `Dimension` object itself. This can be useful, because it
        allows to check for the dimension of an object by checking its ``dim``
        attribute -- this will return a `Dimension` object for `Quantity`,
        `Unit` and `Dimension`.
        '''
        return self

    #### REPRESENTATION ####
    def _str_representation(self, python_code=False):
        """
        String representation in basic SI units, or ``"1"`` for dimensionless.
        Use ``python_code=False`` for display purposes and ``True`` for valid
        Python code.
        """

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

    def _latex(self, *args):
        parts = []
        for i in xrange(len(self._dims)):
            if self._dims[i]:
                s = _ilabel[i]
                if self._dims[i] != 1:
                    s += '^{%s}' % str(self._dims[i])
                parts.append(s)
        s = "\,".join(parts)
        if not len(s):
            return "1"
        return s.strip()

    def _repr_latex(self):
        return '$%s$' % latex(self)

    def __repr__(self):
        return self._str_representation(python_code=True)

    def __str__(self):
        return self._str_representation(python_code=False)

    #### ARITHMETIC ####
    # Note that none of the dimension arithmetic objects do sanity checking
    # on their inputs, although most will throw an exception if you pass the
    # wrong sort of input
    def __mul__(self, value):
        return get_or_create_dimension([x + y for x, y in
                                        itertools.izip(self._dims, value._dims)])

    def __div__(self, value):
        return get_or_create_dimension([x - y for x, y in
                                        itertools.izip(self._dims, value._dims)])

    def __truediv__(self, value):
        return self.__div__(value)

    def __pow__(self, value):
        value = np.array(value, copy=False)
        if value.size > 1:
            raise TypeError('Too many exponents')
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
        try:
            return np.allclose(self._dims, value._dims)
        except AttributeError:
            # Only compare equal to another Dimensions object
            return False

    def __ne__(self, value):
        return not self.__eq__(value)

    def __hash__(self):
        return hash(self._dims)

    #### MAKE DIMENSION PICKABLE ####
    def __getstate__(self):
        return self._dims

    def __setstate__(self, state):
        self._dims = state

    ### Dimension objects are singletons and deepcopy is therefore not necessary
    def __deepcopy__(self, memodict):
        return self


#: The singleton object for dimensionless Dimensions.
DIMENSIONLESS = Dimension((0, 0, 0, 0, 0, 0, 0))

_dimensions = {(0, 0, 0, 0, 0, 0, 0): DIMENSIONLESS}


def get_or_create_dimension(*args, **kwds):
    """
    Create a new Dimension object or get a reference to an existing one.
    This function takes care of only creating new objects if they were not
    created before and otherwise returning a reference to an existing object.
    This allows to compare dimensions very efficiently using ``is``.

    Parameters
    ----------
    args : sequence of `float`
        A sequence with the indices of the 7 elements of an SI dimension.
    kwds : keyword arguments
        a sequence of ``keyword=value`` pairs where the keywords are the names of
        the SI dimensions, or the standard unit.

    Examples
    --------
    The following are all definitions of the dimensions of force

    >>> from brian2 import *
    >>> get_or_create_dimension(length=1, mass=1, time=-2)
    metre * kilogram * second ** -2
    >>> get_or_create_dimension(m=1, kg=1, s=-2)
    metre * kilogram * second ** -2
    >>> get_or_create_dimension([1, 1, -2, 0, 0, 0, 0])
    metre * kilogram * second ** -2

    Notes
    -----
    The 7 units are (in order):

    Length, Mass, Time, Electric Current, Temperature,
    Quantity of Substance, Luminosity

    and can be referred to either by these names or their SI unit names,
    e.g. length, metre, and m all refer to the same thing here.
    """
    if len(args):
        # initialisation by list
        dims = args[0]
        try:
            if len(dims) != 7:
                raise TypeError()
        except TypeError:
            raise TypeError('Need a sequence of exactly 7 items')
    else:
        # initialisation by keywords
        dims = [0, 0, 0, 0, 0, 0, 0]
        for k in kwds.keys():
            # _di stores the index of the dimension with name 'k'
            dims[_di[k]] = kwds[k]

    dims = tuple(dims)

    # check whether this Dimension object has already been created
    if dims in _dimensions:
        return _dimensions[dims]
    else:
        new_dim = Dimension(dims)
        _dimensions[dims] = new_dim
        return new_dim


class DimensionMismatchError(Exception):
    """
    Exception class for attempted operations with inconsistent dimensions.

    For example, ``3*mvolt + 2*amp`` raises this exception. The purpose of this
    class is to help catch errors based on incorrect units. The exception will
    print a representation of the dimensions of the two inconsistent objects
    that were operated on.

    Parameters
    ----------
    description : ``str``
        A description of the type of operation being performed, e.g. Addition,
        Multiplication, etc.
    dims : `Dimension`
        The physical dimensions of the objects involved in the operation, any
        number of them is possible
    """
    def __init__(self, description, *dims):
        # Call the base class constructor to make Exception pickable, see:
        # http://bugs.python.org/issue1692335
        Exception.__init__(self, description, *dims)
        self.dims = dims
        self.desc = description

    def __repr__(self):
        dims_repr = [repr(dim) for dim in self.dims]
        return '%s(%r, %s)' % (self.__class__.__name__,
                               self.desc, ', '.join(dims_repr))

    def __str__(self):
        s = self.desc
        if len(self.dims) == 0:
            pass
        elif len(self.dims) == 1:
            s += ' (unit is ' + get_unit_for_display(self.dims[0])
        elif len(self.dims) == 2:
            d1, d2 = self.dims
            s += ' (units are %s and %s' % (get_unit_for_display(d1),
                                            get_unit_for_display(d2))
        else:
            s += (' (units are ' +
                  ' '.join(['(' + get_unit_for_display(d) + ')'
                            for d in self.dims]))
        if len(self.dims):
            s += ').'
        return s


def is_scalar_type(obj):
    """
    Tells you if the object is a 1d number type.

    Parameters
    ----------
    obj : `object`
        The object to check.

    Returns
    -------
    scalar : `bool`
        ``True`` if `obj` is a scalar that can be interpreted as a
        dimensionless `Quantity`.
    """
    try:
        return obj.ndim == 0
    except AttributeError:
        return np.isscalar(obj) and not isinstance(obj, basestring)


def get_dimensions(obj):
    """
    Return the dimensions of any object that has them.

    Slightly more general than `Quantity.dimensions` because it will
    return `DIMENSIONLESS` if the object is of number type but not a `Quantity`
    (e.g. a `float` or `int`).

    Parameters
    ----------
    obj : `object`
        The object to check.

    Returns
    -------
    dim: `Dimension`
        The physical dimensions of the `obj`.
    """
    try:
        return obj.dim
    except AttributeError:
        # The following is not very pretty, but it will avoid the costly
        # isinstance check for the common types
        if (type(obj) in [int, long, float, np.int32, np.int64,
                          np.float32, np.float64, np.ndarray] or
                isinstance(obj, (numbers.Number, np.number, np.ndarray))):
            return DIMENSIONLESS
        try:
            return Quantity(obj).dim
        except TypeError:
            raise TypeError('Object of type %s does not have dimensions' %
                            type(obj))


def is_dimensionless(obj):
    """
    Test if a value is dimensionless or not.

    Parameters
    ----------
    obj : `object`
        The object to check.

    Returns
    -------
    dimensionless : `bool`
        ``True`` if `obj` is dimensionless.
    """
    return get_dimensions(obj) is DIMENSIONLESS


def have_same_dimensions(obj1, obj2):
    """Test if two values have the same dimensions.

    Parameters
    ----------
    obj1, obj2 : {`Quantity`, array-like, number}
        The values of which to compare the dimensions.

    Returns
    -------
    same : `bool`
        ``True`` if `obj1` and `obj2` have the same dimensions.
    """

    if not unit_checking:
        return True  # ignore units when unit checking is disabled

    # If dimensions are consistently created using get_or_create_dimensions,
    # the fast "is" comparison should always return the correct result.
    # To be safe, we also do an equals comparison in case it fails. This
    # should only add a small amount of unnecessary computation for cases in
    # which this function returns False which very likely leads to a
    # DimensionMismatchError anyway.
    dim1 = get_dimensions(obj1)
    dim2 = get_dimensions(obj2)
    return (dim1 is dim2) or (dim1 == dim2) or dim1 is None or dim2 is None


def in_unit(x, u, precision=None):
    """
    Display a value in a certain unit with a given precision.

    Parameters
    ----------
    x : {`Quantity`, array-like, number}
        The value to display
    u : {`Quantity`, `Unit`}
        The unit to display the value `x` in.
    precision : `int`, optional    
        The number of digits of precision (in the given unit, see Examples).
        If no value is given, numpy's `get_printoptions` value is used.

    Returns
    -------
    s : `str`
        A string representation of `x` in units of `u`.

    Examples
    --------
    >>> from brian2 import *
    >>> in_unit(3 * volt, mvolt)
    '3000. mV'
    >>> in_unit(123123 * msecond, second, 2)
    '123.12 s'
    >>> in_unit(10 * uA/cm**2, nA/um**2)
    '1.00000000e-04 nA/um^2'
    >>> in_unit(10 * mV, ohm * amp)
    '0.01 ohm A'
    >>> in_unit(10 * nS, ohm) # doctest: +NORMALIZE_WHITESPACE
    ...                       # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    DimensionMismatchError: Non-matching unit for method "in_unit",
    dimensions were (m^-2 kg^-1 s^3 A^2) (m^2 kg s^-3 A^-2)

    See Also
    --------
    Quantity.in_unit    
    """
    if is_dimensionless(x):
        fail_for_dimension_mismatch(x, u,   
                                    'Non-matching unit for function '
                                    '"in_unit"')        
        return str(np.array(x / u, copy=False))
    else:
        return x.in_unit(u, precision=precision)


def in_best_unit(x, precision=None):
    """
    Represent the value in the "best" unit.

    Parameters
    ----------
    x : {`Quantity`, array-like, number}
        The value to display
    precision : `int`, optional
        The number of digits of precision (in the best unit, see Examples).
        If no value is given, numpy's `get_printoptions` value is used.            

    Returns
    -------
    representation : `str`
        A string representation of this `Quantity`.

    Examples
    --------
    >>> from brian2.units import *
    >>> in_best_unit(0.00123456 * volt)
    '1.23456 mV'
    >>> in_best_unit(0.00123456 * volt, 2)
    '1.23 mV'
    >>> in_best_unit(0.123456)
    '0.123456'
    >>> in_best_unit(0.123456, 2)
    '0.12'

    See Also
    --------
    Quantity.in_best_unit
    """
    if is_dimensionless(x):
        if precision is None:
            precision = np.get_printoptions()['precision']
        return str(np.round(x, precision))
    
    u = x.get_best_unit()
    return x.in_unit(u, precision=precision)


def quantity_with_dimensions(floatval, dims):
    '''
    Create a new `Quantity` with the given dimensions. Calls
    `get_or_create_dimensions` with the dimension tuple of the `dims`
    argument to make sure that unpickling (which calls this function) does not
    accidentally create new Dimension objects which should instead refer to
    existing ones.

    Parameters
    ----------
    floatval : `float`
        The floating point value of the quantity.
    dims : `Dimension`
        The physical dimensions of the quantity.

    Returns
    -------
    q : `Quantity`
        A quantity with the given dimensions.

    Examples
    --------
    >>> from brian2 import *
    >>> quantity_with_dimensions(0.001, volt.dim)
    1. * mvolt

    See Also
    --------
    get_or_create_dimensions
    '''
    return Quantity(floatval, get_or_create_dimension(dims._dims))


class Quantity(np.ndarray, object):
    """
    A number with an associated physical dimension. In most cases, it is not
    necessary to create a Quantity object by hand, instead use multiplication
    and division of numbers with the constant unit names ``second``,
    ``kilogram``, etc.

    Notes
    -----
    The `Quantity` class defines arithmetic operations which check for
    consistency of dimensions and raise the DimensionMismatchError exception
    if they are inconsistent. It also defines default and other representations
    for a number for printing purposes.

    See the documentation on the Unit class for more details
    about the available unit names like mvolt, etc.

    *Casting rules*

    The rules that define the casting operations for
    Quantity object are:

    1. Quantity op Quantity = Quantity
       Performs dimension checking if appropriate
    2. (Scalar or Array) op Quantity = Quantity
       Assumes that the scalar or array is dimensionless

    There is one exception to the above rule, the number ``0`` is interpreted
    as having "any dimension".

    Examples
    --------
    >>> from brian2 import *
    >>> I = 3 * amp # I is a Quantity object
    >>> R = 2 * ohm # same for R
    >>> I * R
    6. * volt
    >>> (I * R).in_unit(mvolt)
    '6000. mV'
    >>> (I * R) / mvolt
    6000.0
    >>> X = I + R  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    DimensionMismatchError: Addition, dimensions were (A) (m^2 kg s^-3 A^-2)
    >>> Is = np.array([1, 2, 3]) * amp
    >>> Is * R
    array([ 2.,  4.,  6.]) * volt
    >>> np.asarray(Is * R) # gets rid of units
    array([ 2.,  4.,  6.])

    See also
    --------
    Unit

    Attributes
    ----------
    dimensions
    is_dimensionless
    dim : Dimensions
        The physical dimensions of this quantity.

    Methods
    -------
    with_dimensions
    has_same_dimensions
    in_unit
    in_best_unit
    """
    __slots__ = ["dim"]

    __array_priority__ = 1000

    #==========================================================================
    # Construction and handling of numpy ufuncs
    #==========================================================================
    def __new__(cls, arr, dim=None, dtype=None, copy=False, force_quantity=False):

        # Do not create dimensionless quantities, use pure numpy arrays instead
        if dim is DIMENSIONLESS and not force_quantity:
            arr = np.array(arr, dtype=dtype, copy=copy)
            if arr.shape == ():
                # For scalar values, return a simple Python object instead of
                # a numpy scalar
                return arr.item()
            return arr

        # All np.ndarray subclasses need something like this, see
        # http://www.scipy.org/Subclasses
        subarr = np.array(arr, dtype=dtype, copy=copy).view(cls)

        # We only want numerical datatypes
        if not np.issubclass_(np.dtype(subarr.dtype).type,
                              (np.number, np.bool_)):
            raise TypeError('Quantities can only be created from numerical data.')

        # If a dimension is given, force this dimension
        if dim is not None:
            subarr.dim = dim
            return subarr

        # Use the given dimension or the dimension of the given array (if any)
        try:
            subarr.dim = arr.dim
        except AttributeError:
            if not isinstance(arr, (np.ndarray, np.number, numbers.Number)):
                # check whether it is an iterable containing Quantity objects
                try:
                    is_quantity = [isinstance(x, Quantity)
                                   for x in _flatten(arr)]
                except TypeError:
                    # Not iterable
                    is_quantity = [False]
                if len(is_quantity) == 0:
                    # Empty list
                    subarr.dim = DIMENSIONLESS
                elif all(is_quantity):
                    dims = [x.dim for x in _flatten(arr)]
                    one_dim = dims[0]
                    for d in dims:
                        if d != one_dim:
                            raise DimensionMismatchError('Mixing quantities '
                                                         'with different '
                                                         'dimensions is not '
                                                         'allowed',
                                                         d, one_dim)
                    subarr.dim = dims[0]
                elif any(is_quantity):
                    raise TypeError('Mixing quantities and non-quantities is '
                                    'not allowed.')

        return subarr

    def __array_finalize__(self, orig):
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
            fail_for_dimension_mismatch(args[0], args[1],
                                        error_message=('Cannot calculate '
                                                       '{val1} %s {val2}, the '
                                                       'units do not '
                                                       'match') % uf.__name__,
                                        val1=args[0], val2=args[1])
        elif uf.__name__ in UFUNCS_DIMENSIONLESS:
            # Ok if argument is dimensionless
            fail_for_dimension_mismatch(args[0],
                                        error_message=('%s expects a '
                                                       'dimensionless argument '
                                                       'but got '
                                                       '{value}') % uf.__name__,
                                        value=args[0])
        elif uf.__name__ in UFUNCS_DIMENSIONLESS_TWOARGS:
            # Ok if both arguments are dimensionless
            fail_for_dimension_mismatch(args[0],
                                        error_message=('Both arguments for '
                                                       '"%s" should be '
                                                       'dimensionless but '
                                                       'first argument was '
                                                       '{value}') % uf.__name__,
                                        value=args[0])
            fail_for_dimension_mismatch(args[1],
                                        error_message=('Both arguments for '
                                                       '"%s" should be '
                                                       'dimensionless but '
                                                       'second argument was '
                                                       '{value}') % uf.__name__,
                                        value=args[1])
        elif uf.__name__ == 'power':
            fail_for_dimension_mismatch(args[1],
                                        error_message='The exponent for a '
                                                      'power operation has to '
                                                      'be dimensionless but '
                                                      'was {value}',
                                        value=args[1])
            if np.array(args[1], copy=False).size != 1:
                raise TypeError('Only length-1 arrays can be used as an '
                                'exponent for quantities.')
        elif uf.__name__ in ('sign', 'ones_like'):
            return np.array(array, copy=False)
        else:
            warn("Unknown ufunc '%s' in __array_prepare__" % uf.__name__)

        return array

    def __array_wrap__(self, array, context=None):
        dim = DIMENSIONLESS

        if not context is None:
            uf, args, _ = context
            if uf.__name__ in (UFUNCS_PRESERVE_DIMENSIONS +
                               UFUNCS_MATCHING_DIMENSIONS):
                dim = self.dim
            elif uf.__name__ in (UFUNCS_DIMENSIONLESS +
                                 UFUNCS_DIMENSIONLESS_TWOARGS):
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
                dim = get_dimensions(args[0]) ** np.array(args[1], copy=False)
            elif uf.__name__ == 'square':
                dim = self.dim ** 2
            elif uf.__name__ in ('divide', 'true_divide', 'floor_divide'):
                dim = get_dimensions(args[0]) / get_dimensions(args[1])
            elif uf.__name__ == 'reciprocal':
                dim = get_dimensions(args[0]) ** -1
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

    def __deepcopy__(self, memo):
        return Quantity(self, copy=True)

#==============================================================================
# Quantity-specific functions (not existing in ndarray)
#==============================================================================
    @staticmethod
    def with_dimensions(value, *args, **keywords):
        """
        Create a `Quantity` object with dim.

        Parameters
        ----------
        value : {array_like, number}
            The value of the dimension
        args : {`Dimension`, sequence of float}
            Either a single argument (a `Dimension`) or a sequence of 7 values.
        kwds
            Keywords defining the dim, see `Dimension` for details.

        Returns
        -------
        q : `Quantity`
            A `Quantity` object with the given dim

        Examples
        --------
        All of these define an equivalent `Quantity` object:

        >>> from brian2 import *
        >>> Quantity.with_dimensions(2, get_or_create_dimension(length=1))
        2. * metre
        >>> Quantity.with_dimensions(2, length=1)
        2. * metre
        >>> 2 * metre
        2. * metre
        """
        if len(args) and isinstance(args[0], Dimension):
            dimensions = args[0]
        else:
            dimensions = get_or_create_dimension(*args, **keywords)
        return Quantity(value, dim=dimensions)

    ### ATTRIBUTES ###
    is_dimensionless = property(lambda self: self.dim.is_dimensionless,
                                doc='Whether this is a dimensionless quantity.')

    @property
    def dimensions(self):
        '''
        The physical dimensions of this quantity.
        '''
        return self.dim

    @dimensions.setter
    def dimensions(self, dim):
        self.dim = dim

    #### METHODS ####

    def has_same_dimensions(self, other):
        """
        Return whether this object has the same dimensions as another.

        Parameters
        ----------
        other : {`Quantity`, array-like, number}
            The object to compare the dimensions against.

        Returns
        -------
        same : `bool`
            ``True`` if `other` has the same dimensions.
        """
        if not unit_checking:
            return True  # ignore units if unit checking is disabled

        other_dim = get_dimensions(other)
        return (self.dim is other_dim) or (self.dim == other_dim) 

    def in_unit(self, u, precision=None, python_code=False):
        """
        Represent the quantity in a given unit. If `python_code` is ``True``,
        this will return valid python code, i.e. a string like ``5.0 * um ** 2``
        instead of ``5.0 um^2``

        Parameters
        ----------
        u : {`Quantity`, `Unit`}
            The unit in which to show the quantity.
        precision : `int`, optional
            The number of digits of precision (in the given unit, see Examples).
            If no value is given, numpy's `get_printoptions` value is used.
        python_code : `bool`, optional
            Whether to return valid python code (``True``) or a human readable
            string (``False``, the default).

        Returns
        -------
        s : `str`
            String representation of the object in unit `u`.

        Examples
        --------
        >>> from brian2.units import *
        >>> from brian2.units.stdunits import *
        >>> x = 25.123456 * mV
        >>> x.in_unit(volt)
        '0.02512346 V'
        >>> x.in_unit(volt, 3)
        '0.025 V'
        >>> x.in_unit(mV, 3)
        '25.123 mV'

        See Also
        --------
        in_unit
        """

        fail_for_dimension_mismatch(self, u,
                                    'Non-matching unit for method "in_unit"')
        
        value = np.array(self / u, copy=False)
        # numpy uses the printoptions setting only in arrays, not in array
        # scalars, so we use this hackish way of turning the scalar first into
        # an array, then removing the square brackets from the output
        if value.shape == ():
            s = np.array_str(np.array([value]), precision=precision)
            s = s.replace('[', '').replace(']', '').strip()
        else:
            if python_code:
                s = np.array_repr(value, precision=precision)
            else:
                s = np.array_str(value, precision=precision)

        if not u.is_dimensionless:
            if isinstance(u, Unit):
                if python_code:
                    s += ' * ' + repr(u)
                else:
                    s += ' ' + str(u)
            else:
                if python_code:
                    s += ' * ' + repr(u.dim)
                else:
                    s += ' ' + str(u.dim)
        elif python_code == True:  # Make a quantity without unit recognisable
            return '%s(%s)' % (self.__class__.__name__, s.strip())
        return s.strip()

    def get_best_unit(self, *regs):
        """
        Return the best unit for this `Quantity`.

        Parameters
        ----------
        regs : any number of `UnitRegistry` objects
            The registries that are searched for units. If none are provided, it
            will check the standard, user and additional unit registers in turn.

        Returns
        -------
            u : `Quantity` or `Unit`
                The best-fitting unit for the quantity `x`.
        """
        if self.is_dimensionless:
            return Unit(1)
        if len(regs):
            for r in regs:
                try:
                    return r[self]
                except KeyError:
                    pass
            return Quantity(1, self.dim)
        else:
            return self.get_best_unit(standard_unit_register,
                                      user_unit_register,
                                      additional_unit_register)

    def _get_best_unit(self, *regs):
        warn('Quantity._get_best_unit has been renamed to '
             'Quantity.get_best_unit.', VisibleDeprecationWarning)
        return self.get_best_unit(*regs)

    def in_best_unit(self, precision=None, python_code=False, *regs):
        """
        Represent the quantity in the "best" unit.

        Parameters
        ----------
        python_code : `bool`, optional
            If set to ``False`` (the default), will return a string like
            ``5.0 um^2`` which is not a valid Python expression. If set to
            ``True``, it will return ``5.0 * um ** 2`` instead.
        precision : `int`, optional
            The number of digits of precision (in the best unit, see
            Examples). If no value is given, numpy's
            `get_printoptions` value is used.            
        regs : `UnitRegistry` objects
            The registries where to search for units. If none are given, the
            standard, user-defined and additional registries are searched in
            that order.

        Returns
        -------
        representation : `str`
            A string representation of this `Quantity`.

        Examples
        --------
        >>> from brian2.units import *

        >>> x = 0.00123456 * volt

        >>> x.in_best_unit()
        '1.23456 mV'

        >>> x.in_best_unit(3)
        '1.235 mV'

        See Also
        --------
        in_best_unit
        """
        u = self.get_best_unit(*regs)
        return self.in_unit(u, precision=precision, python_code=python_code)

#==============================================================================
# Overwritten ndarray methods
#==============================================================================

    #### Setting/getting items ####
    def __getitem__(self, key):
        ''' Overwritten to assure that single elements (i.e., indexed with a
        single integer or a tuple of integers) retain their unit.
        '''
        return Quantity(np.ndarray.__getitem__(self, key), self.dim)

    def __getslice__(self, start, end):
        return self.__getitem__(slice(start, end))

    def __setitem__(self, key, value):
        fail_for_dimension_mismatch(self, value,
                                    'Inconsistent units in assignment')
        return super(Quantity, self).__setitem__(key, value)

    def __setslice__(self, start, end, value):
        return self.__setitem__(slice(start, end), value)

    #### ARITHMETIC ####
    def _binary_operation(self, other, operation,
                          dim_operation=lambda a, b: a, fail_for_mismatch=False,
                          operator_str=None, inplace=False):
        '''
        General implementation for binary operations.

        Parameters
        ----------
        other : {`Quantity`, `ndarray`, scalar}
            The object with which the operation should be performed.
        operation : function of two variables
            The function with which the two objects are combined. For example,
            `operator.mul` for a multiplication.
        dim_operation : function of two variables, optional
            The function with which the dimension of the resulting object is
            calculated (as a function of the dimensions of the two involved
            objects). For example, `operator.mul` for a multiplication. If not
            specified, the dimensions of `self` are used for the resulting
            object.
        fail_for_mismatch : bool, optional
            Whether to fail for a dimension mismatch between `self` and `other`
            (defaults to ``False``)
        operator_str : str, optional
            The string to use for the operator in an error message.
        inplace: bool, optional
            Whether to do the operation in-place (defaults to ``False``).

        Notes
        -----
        For in-place operations on scalar values, a copy of the original object
        is returned, i.e. it rather works like a fundamental Python type and
        not like a numpy array scalar, preventing weird effects when a reference
        to the same value was stored in another variable. See github issue #469.
        '''
        other_dim = None

        if fail_for_mismatch:
            if inplace:
                message = ('Cannot calculate ... %s {value}, units do not '
                           'match') % operator_str
                _, other_dim = fail_for_dimension_mismatch(self, other,
                                                           message, value=other)
            else:
                message = ('Cannot calculate {value1} %s {value2}, units do not '
                           'match') % operator_str
                _, other_dim = fail_for_dimension_mismatch(self, other, message,
                                                           value1=self,
                                                           value2=other)

        if other_dim is None:
            other_dim = get_dimensions(other)

        if inplace:
            if self.shape == ():
                self_value = Quantity(self, copy=True)
            else:
                self_value = self
            operation(self_value, other)
            self_value.dim = dim_operation(self.dim, other_dim)
            return self_value
        else:
            newdims = dim_operation(self.dim, other_dim)
            self_arr = np.array(self, copy=False)
            other_arr = np.array(other, copy=False)
            result = operation(self_arr, other_arr)
            return Quantity(result, newdims)

    def __mul__(self, other):
        return self._binary_operation(other, operator.mul, operator.mul)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        return self._binary_operation(other, np.ndarray.__imul__, operator.mul,
                                      inplace=True)

    def __div__(self, other):
        return self._binary_operation(other, operator.truediv, operator.truediv)

    def __truediv__(self, other):
        return self.__div__(other)

    def __rdiv__(self, other):
        # division with swapped arguments
        rdiv = lambda a, b: operator.truediv(b, a)
        return self._binary_operation(other, rdiv, rdiv)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __idiv__(self, other):
        return self._binary_operation(other, np.ndarray.__itruediv__,
                                      operator.truediv, inplace=True)

    def __itruediv__(self, other):
        return self._binary_operation(other, np.ndarray.__itruediv__,
                                      operator.truediv, inplace=True)

    def __mod__(self, other):
        return self._binary_operation(other, operator.mod,
                                      fail_for_mismatch=True,
                                      operator_str=r'%')

    def __add__(self, other):
        return self._binary_operation(other, operator.add,
                                      fail_for_mismatch=True,
                                      operator_str='+')

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        return self._binary_operation(other, np.ndarray.__iadd__,
                                      fail_for_mismatch=True,
                                      operator_str='+=',
                                      inplace=True)

    def __sub__(self, other):
        return self._binary_operation(other, operator.sub,
                                      fail_for_mismatch=True,
                                      operator_str='-')

    def __rsub__(self, other):
        # We allow operations with 0 even for dimension mismatches, e.g.
        # 0 - 3*mV is allowed. In this case, the 0 is not represented by a
        # Quantity object so we cannot simply call Quantity.__sub__
        if ((not isinstance(other, Quantity) or other.dim is DIMENSIONLESS) and
                np.all(other == 0)):
            return self.__neg__()
        else:
            return Quantity(other, copy=False, force_quantity=True).__sub__(self)

    def __isub__(self, other):
        return self._binary_operation(other, np.ndarray.__isub__,
                                      fail_for_mismatch=True,
                                      operator_str='-=',
                                      inplace=True)
    def __pow__(self, other):
        if isinstance(other, np.ndarray) or is_scalar_type(other):
            fail_for_dimension_mismatch(other,
                                        error_message='Cannot calculate '
                                                      '{base} ** {exponent}, '
                                                      'the exponent has to be '
                                                      'dimensionless',
                                        base=self, exponent=other)
            other = np.array(other, copy=False)
            return Quantity(np.array(self, copy=False)**other,
                            self.dim**other)
        else:
            return NotImplemented

    def __rpow__(self, other):
        if self.is_dimensionless:
            if isinstance(other, np.ndarray) or isinstance(other, np.ndarray):
                new_array = np.array(other, copy=False)**np.array(self,
                                                                  copy=False)
                return Quantity(new_array, DIMENSIONLESS)
            else:
                return NotImplemented
        else:
            raise DimensionMismatchError(('Cannot calculate '
                                          '{base} ** {exponent}, the '
                                          'exponent has to be '
                                          'dimensionless').format(base=_short_str(other),
                                                                  exponent=_short_str(self)),
                                         self.dim)

    def __ipow__(self, other):
        if isinstance(other, np.ndarray) or is_scalar_type(other):
            fail_for_dimension_mismatch(other,
                                        error_message='Cannot calculate '
                                                      '... **= {exponent}, '
                                                      'the exponent has to be '
                                                      'dimensionless',
                                        exponent=other)
            other = np.array(other, copy=False)
            super(Quantity, self).__ipow__(other)
            self.dim = self.dim ** other
            return self
        else:
            return NotImplemented

    def __neg__(self):
        return Quantity(-np.array(self, copy=False), self.dim)

    def __pos__(self):
        return self

    def __abs__(self):
        return Quantity(abs(np.array(self, copy=False)), self.dim)

    def tolist(self):
        '''
        Convert the array into a list.

        Returns
        -------
        l : list of `Quantity`
            A (possibly nested) list equivalent to the original array.
        '''
        def replace_with_quantity(seq, dim):
            '''
            Replace all the elements in the list with an equivalent `Quantity`
            with the given `dim`.
            '''
            # No recursion needed for single values
            if not isinstance(seq, list):
                return Quantity(seq, dim)

            def top_replace(s):
                '''
                Recursivley descend into the list.
                '''
                for i in s:
                    if not isinstance(i, list):
                        yield Quantity(i, dim)
                    else:
                        yield type(i)(top_replace(i))

            return type(seq)(top_replace(seq))

        return replace_with_quantity(np.array(self, copy=False).tolist(),
                                     self.dim)

    #### COMPARISONS ####
    def _comparison(self, other, operator_str, operation):
        is_scalar = is_scalar_type(other)
        if not is_scalar and not isinstance(other, np.ndarray):
            return NotImplemented
        if not is_scalar or not np.isinf(other):
                message = ('Cannot perform comparison {value1} %s {value2}, '
                           'units do not match') % operator_str
                fail_for_dimension_mismatch(self, other, message,
                                            value1=self, value2=other)
        return operation(np.array(self, copy=False),
                         np.array(other, copy=False))

    def __lt__(self, other):
        return self._comparison(other, '<', operator.lt)

    def __le__(self, other):
        return self._comparison(other, '<=', operator.le)

    def __gt__(self, other):
        return self._comparison(other, '>', operator.gt)

    def __ge__(self, other):
        return self._comparison(other, '>=', operator.ge)

    def __eq__(self, other):
        return self._comparison(other, '==', operator.eq)

    def __ne__(self, other):
        return self._comparison(other, '!=', operator.ne)

    #### MAKE QUANTITY PICKABLE ####
    def __reduce__(self):
        return quantity_with_dimensions, (np.array(self, copy=False), self.dim)

    #### REPRESENTATION ####
    def __repr__(self):
        return self.in_best_unit(python_code=True)

    # TODO: Use sympy's _latex method, then latex(unit) should work
    def _latex(self, expr):
        from sympy import Matrix
        best_unit = self.get_best_unit()
        if isinstance(best_unit, Unit):
            best_unit_latex = latex(best_unit)
        else: # A quantity
            best_unit_latex = latex(best_unit.dimensions)
        unitless = np.array(self / best_unit, copy=False)
        if unitless.ndim == 0:
            sympy_quantity = np.float(unitless)
        else:
            sympy_quantity = Matrix(unitless)
        return latex(sympy_quantity) + '\,' + best_unit_latex

    def _repr_latex_(self):
        return  '$' + latex(self) + '$'

    def __str__(self):
        return self.in_best_unit()

    #### Mathematic methods ####

    cumsum = wrap_function_keep_dimensions(np.ndarray.cumsum)
    diagonal = wrap_function_keep_dimensions(np.ndarray.diagonal)
    max = wrap_function_keep_dimensions(np.ndarray.max)
    mean = wrap_function_keep_dimensions(np.ndarray.mean)
    min = wrap_function_keep_dimensions(np.ndarray.min)
    ptp = wrap_function_keep_dimensions(np.ndarray.ptp)

    # To work around an issue in matplotlib 1.3.1 (see
    # https://github.com/matplotlib/matplotlib/pull/2591), we make `ravel`
    # return a unitless array and emit a warning explaining the issue.
    use_matplotlib_units_fix = False
    try:
        import matplotlib
        if matplotlib.__version__ == '1.3.1':
            use_matplotlib_units_fix = True
    except ImportError:
        pass

    if use_matplotlib_units_fix:
        def ravel(self, *args, **kwds):
            # Note that we don't use Brian's logging system here as we don't want
            # the unit system to depend on other parts of Brian
            warn(('As a workaround for a bug in matplotlib 1.3.1, calling '
                  '"ravel()" on a quantity will return unit-less values. If you '
                  'get this warning during plotting, consider removing the units '
                  'before plotting, e.g. by dividing by the unit. If you are '
                  'explicitly calling "ravel()", consider using "flatten()" '
                  'instead.'))
            return np.array(self, copy=False).ravel(*args, **kwds)

        ravel._arg_units = [None]
        ravel._return_unit = 1
        ravel.__name__ = np.ndarray.ravel.__name__
        ravel.__doc__ = np.ndarray.ravel.__doc__
    else:
        ravel = wrap_function_keep_dimensions(np.ndarray.ravel)

    round = wrap_function_keep_dimensions(np.ndarray.round)
    std = wrap_function_keep_dimensions(np.ndarray.std)
    sum = wrap_function_keep_dimensions(np.ndarray.sum)
    trace = wrap_function_keep_dimensions(np.ndarray.trace)
    var = wrap_function_change_dimensions(np.ndarray.var, lambda ar, d: d ** 2)
    all = wrap_function_remove_dimensions(np.ndarray.all)
    any = wrap_function_remove_dimensions(np.ndarray.any)
    nonzero = wrap_function_remove_dimensions(np.ndarray.nonzero)
    argmax = wrap_function_remove_dimensions(np.ndarray.argmax)
    argmin = wrap_function_remove_dimensions(np.ndarray.argmin)
    argsort = wrap_function_remove_dimensions(np.ndarray.argsort)

    def fill(self, values): # pylint: disable=C0111
        fail_for_dimension_mismatch(self, values, 'fill')
        super(Quantity, self).fill(values)
    fill.__doc__ = np.ndarray.fill.__doc__

    def put(self, indices, values, *args, **kwds): # pylint: disable=C0111
        fail_for_dimension_mismatch(self, values, 'fill')
        super(Quantity, self).put(indices, values, *args, **kwds)
    put.__doc__ = np.ndarray.put.__doc__

    def clip(self, a_min, a_max, *args, **kwds): # pylint: disable=C0111
        fail_for_dimension_mismatch(self, a_min, 'clip')
        fail_for_dimension_mismatch(self, a_max, 'clip')
        return Quantity(np.clip(np.array(self, copy=False),
                                np.array(a_min, copy=False),
                                np.array(a_max, copy=False),
                                *args, **kwds),
                        self.dim)
    clip.__doc__ = np.ndarray.clip.__doc__

    def dot(self, other, **kwds): # pylint: disable=C0111
        return Quantity(np.array(self).dot(np.array(other),
                                           **kwds),
                        self.dim*get_dimensions(other))
    dot.__doc__ = np.ndarray.dot.__doc__

    def searchsorted(self, v, **kwds): # pylint: disable=C0111
        fail_for_dimension_mismatch(self, v, 'searchsorted')
        return super(Quantity, self).searchsorted(np.array(v, copy=False),
                                                  **kwds)
    searchsorted.__doc__ = np.ndarray.searchsorted.__doc__

    def prod(self, *args, **kwds): # pylint: disable=C0111
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
        return Quantity(np.array(prod_result, copy=False),
                        self.dim ** dim_exponent)
    prod.__doc__ = np.ndarray.prod.__doc__

    def cumprod(self, *args, **kwds):  # pylint: disable=C0111
        if not self.is_dimensionless:
            raise TypeError('cumprod over array elements on quantities '
                            'with dimensions is not possible.')
        return Quantity(np.array(self, copy=False).cumprod(*args, **kwds))
    cumprod.__doc__ = np.ndarray.cumprod.__doc__


class Unit(Quantity):
    r'''
    A physical unit.

    Normally, you do not need to worry about the implementation of
    units. They are derived from the `Quantity` object with
    some additional information (name and string representation).
    
    Basically, a unit is just a number with given dimensions, e.g.
    mvolt = 0.001 with the dimensions of voltage. The units module
    defines a large number of standard units, and you can also define
    your own (see below).

    The unit class also keeps track of various things that were used
    to define it so as to generate a nice string representation of it.
    See below.

    When creating scaled units, you can use the following prefixes:

     ======     ======  ==============
     Factor     Name    Prefix
     ======     ======  ==============
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
     ======     ======  ==============

   **Defining your own**

    It can be useful to define your own units for printing
    purposes. So for example, to define the newton metre, you
    write

    >>> from brian2 import *
    >>> from brian2.units.allunits import newton
    >>> Nm = newton * metre

    You can then do

    >>> (1*Nm).in_unit(Nm)
    '1. N m'

    New "compound units", i.e. units that are composed of other units will be
    automatically registered and from then on used for display. For example,
    imagine you define total conductance for a membrane, and the total area of
    that membrane:

    >>> conductance = 10.*nS
    >>> area = 20000*um**2

    If you now ask for the conductance density, you will get an "ugly" display
    in basic SI dimensions, as Brian does not know of a corresponding unit:

    >>> conductance/area
    0.5 * metre ** -4 * kilogram ** -1 * second ** 3 * amp ** 2

    By using an appropriate unit once, it will be registered and from then on
    used for display when appropriate:

    >>> usiemens/cm**2
    usiemens / cmetre ** 2
    >>> conductance/area  # same as before, but now Brian nows about uS/cm^2
    50. * usiemens / cmetre ** 2

    Note that user-defined units cannot override the standard units (`volt`,
    `second`, etc.) that are predefined by Brian. For example, the unit
    ``Nm`` has the dimensions "length²·mass/time²", and therefore the same
    dimensions as the standard unit `joule`. The latter will be used for display
    purposes:

    >>> 3*joule
    3. * joule
    >>> 3*Nm
    3. * joule

    '''
    __slots__ = ["dim", "scale", "_dispname", "_name",
                 "_latexname", "iscompound"]

    __array_priority__ = 100

    automatically_register_units = True

    #### CONSTRUCTION ####
    def __new__(cls, arr, dim=None, scale=0, name=None, dispname=None,
                latexname=None, iscompound=False, dtype=None, copy=False):
        if dim is None:
            dim = DIMENSIONLESS
        obj = super(Unit, cls).__new__(cls, arr, dim=dim, dtype=dtype,
                                       copy=copy, force_quantity=True)
        return obj

    def __array_finalize__(self, orig):
        self.dim = getattr(orig, 'dim', DIMENSIONLESS)
        self.scale = getattr(orig, 'scale', 0)
        self._name = getattr(orig, '_name', '')
        self._dispname = getattr(orig, '_dispname', '')
        self._latexname = getattr(orig, '_latexname', '')
        self.iscompound = getattr(orig, '_iscompound', False)
        return self

    def __init__(self, value, dim=None, scale=0, name=None, dispname=None,
                 latexname='', iscompound=False):
        if value != 10.0**scale:
            raise AssertionError('Unit value has to be 10**scale '
                                 '(scale={scale}, value={value})'.format(scale=scale,
                                                                         value=value))
        if dim is None:
            dim = DIMENSIONLESS
        self.dim = dim  #: The Dimensions of this unit

        #: The scale for this unit (as the integer exponent of 10), i.e.
        #: a scale of 3 means 10^3, e.g. for a "k" prefix.
        self.scale = scale
        if name is None:
            if dim is DIMENSIONLESS:
                name = 'Unit(1)'
            else:
                name = repr(dim)
        if dispname is None:
            if dim is DIMENSIONLESS:
                dispname = '1'
            else:
                dispname = str(dim)
        #: The full name of this unit.
        self._name = name
        #: The display name of this unit.
        self._dispname = dispname
        #: A LaTeX expression for the name of this unit.
        self._latexname = latexname
        #: Whether this unit is a combination of other units.
        self.iscompound = iscompound

        if Unit.automatically_register_units:
            register_new_unit(self)

    @staticmethod
    def create(dim, name, dispname, latexname=None, scale=0):
        """
        Create a new named unit.

        Parameters
        ----------
        dim : `Dimension`
            The dimensions of the unit.
        name : `str`
            The full name of the unit, e.g. ``'volt'``
        dispname : `str`
            The display name, e.g. ``'V'``
        latexname : str, optional
            The name as a LaTeX expression (math mode is assumed, do not add
            $ signs or similar), e.g. ``'\omega'``. If no `latexname` is
            specified, `dispname` will be used.
        scale : int, optional
            The scale of this unit as an exponent of 10, e.g. -3 for a unit that
            is 1/1000 of the base scale. Defaults to 0 (i.e. a base unit).

        Returns
        -------
        u : `Unit`
            The new unit.
        """
        name = str(name)
        dispname = str(dispname)
        if latexname is None:
            latexname = r'\mathrm{' + dispname + '}'

        u = Unit(10.0**scale, dim=dim, scale=scale, name=name,
                 dispname=dispname, latexname=latexname)

        return u

    @staticmethod
    def create_scaled_unit(baseunit, scalefactor):
        """
        Create a scaled unit from a base unit.

        Parameters
        ----------
        baseunit : `Unit`
            The unit of which to create a scaled version, e.g. ``volt``,
            ``amp``.
        scalefactor : `str`
            The scaling factor, e.g. ``"m"`` for mvolt, mamp

        Returns
        -------
        u : `Unit`
            The new unit.
        """
        name = scalefactor + baseunit.name
        dispname = scalefactor + baseunit.dispname
        scale = _siprefixes[scalefactor] + baseunit.scale
        if scalefactor == 'u':
            scalefactor = r'\mu'
        latexname = r'\mathrm{' + scalefactor + '}' + r'\,' + baseunit.latexname

        u = Unit(10.0**scale, dim=baseunit.dim,  name=name, dispname=dispname,
                 latexname=latexname, scale=scale)

        return u

    #### METHODS ####
    def set_name(self, name):
        """Sets the name for the unit.

        .. deprecated:: 2.1
            Create a new unit with `Unit.create` instead.
        """
        raise NotImplementedError('Setting the name for a unit after'
                                  'its creation is no longer supported, use'
                                  '"Unit.create" to create a new unit.')


    def set_display_name(self, name):
        """Sets the display name for the unit.

        .. deprecated:: 2.1
            Create a new unit with `Unit.create` instead.
        """
        raise NotImplementedError('Setting the display name for a unit after'
                                  'its creation is no longer supported, use'
                                  '"Unit.create" to create a new unit.')

    def set_latex_name(self, name):
        """Sets the LaTeX name for the unit.

        .. deprecated:: 2.1
            Create a new unit with `Unit.create` instead.
        """
        raise NotImplementedError('Setting the LaTeX name for a unit after'
                                  'its creation is no longer supported, use'
                                  '"Unit.create" to create a new unit.')

    name = property(fget=lambda self: self._name, fset=set_name,
                    doc='The name of the unit')

    dispname = property(fget=lambda self: self._dispname, fset=set_display_name,
                        doc='The display name of the unit')

    latexname = property(fget=lambda self: self._latexname, fset=set_latex_name,
                         doc='The LaTeX name of the unit')

    #### REPRESENTATION ####
    def __repr__(self):
        return self.name

    def __str__(self):
        return self.dispname

    def _latex(self, *args):
        return self.latexname

    def _repr_latex_(self):
        return '$' + latex(self) + '$'

    #### ARITHMETIC ####
    def __mul__(self, other):
        if isinstance(other, Unit):
            name = self.name + " * " + other.name
            dispname = self.dispname + ' ' + other.dispname
            latexname = self.latexname + r'\,' + other.latexname
            scale = self.scale + other.scale
            u = Unit(10.0**scale, dim=self.dim * other.dim, name=name,
                     dispname=dispname, latexname=latexname, iscompound=True,
                     scale=scale)
            return u
        else:
            return super(Unit, self).__mul__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        if isinstance(other, Unit):
            if other.iscompound:
                dispname = '(' + self.dispname + ')'
                name = '(' + self.name + ')'
            else:
                dispname = self.dispname
                name = self.name
            dispname += '/'
            name += ' / '
            if other.iscompound:
                dispname += '(' + other.dispname + ')'
                name += '(' + other.name + ')'
            else:
                dispname += other.dispname
                name += other.name

            latexname = r'\frac{%s}{%s}' % (self.latexname, other.latexname)
            scale = self.scale - other.scale
            u = Unit(10.0**scale, dim=self.dim / other.dim, name=name,
                     dispname=dispname, latexname=latexname, scale=scale,
                     iscompound=True)
            return u
        else:
            return super(Unit, self).__div__(other)

    def __rdiv__(self, other):
        if isinstance(other, Unit):
            return other.__div__(self)
        else:
            try:
                if is_dimensionless(other) and other == 1:
                    return self**-1
            except (ValueError, TypeError, DimensionMismatchError):
                pass
            return super(Unit, self).__rdiv__(other)

    def __pow__(self, other):
        if is_scalar_type(other):
            if self.iscompound:
                dispname = '(' + self.dispname + ')'
                name = '(' + self.name + ')'
                latexname = r'\left(%s\right)' % self.latexname
            else:
                dispname = self.dispname
                name = self.name
                latexname = self.latexname
            dispname += '^' + str(other)
            name += ' ** ' + repr(other)
            latexname += '^{%s}' % latex(other)
            scale = self.scale * other
            u = Unit(10.0**scale, dim=self.dim ** other, name=name,
                     dispname=dispname, latexname=latexname, scale=scale)
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

    def __eq__(self, other):
        if isinstance(other, Unit):
            return other.dim is self.dim and other.scale == self.scale
        else:
            return Quantity.__eq__(self, other)

    def __neq__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.dim, self.scale))


class UnitRegistry(object):
    """
    Stores known units for printing in best units.

    All a user needs to do is to use the `register_new_unit`
    function.

    Default registries:

    The units module defines three registries, the standard units,
    user units, and additional units. Finding best units is done
    by first checking standard, then user, then additional. New
    user units are added by using the `register_new_unit` function.

    Standard units includes all the basic non-compound unit names
    built in to the module, including volt, amp, etc. Additional
    units defines some compound units like newton metre (Nm) etc.

    Methods
    -------
    add
    __getitem__
    """
    def __init__(self):
        self.units = collections.OrderedDict()
        self.units_for_dimensions = collections.defaultdict(dict)

    def add(self, u):
        """Add a unit to the registry
        """
        self.units[repr(u)] = u
        self.units_for_dimensions[u.dim][float(u)] = u

    def __getitem__(self, x):
        """Returns the best unit for quantity x

        The algorithm is to consider the value:

        m=abs(x/u)

        for all matching units u. We select the unit where this ratio is the
        closest to 10 (if it is an array with several values, we select the
        unit where the deviations from that are the smallest. More precisely,
        the unit that minimizes the sum of (log10(m)-1)**2 over all entries).
        """
        matching = self.units_for_dimensions.get(x.dim, {})
        if len(matching) == 0:
            raise KeyError("Unit not found in registry.")

        matching_values = np.array(matching.keys(), copy=False)
        print_opts = np.get_printoptions()
        edgeitems, threshold = print_opts['edgeitems'], print_opts['threshold']
        if x.size > threshold:
            # Only care about optimizing the units for the values that will
            # actually be shown later
            # The code looks a bit complex, but should return the same numbers
            # that are shown by numpy's string conversion
            slices = []
            for shape in x.shape:
                if shape > 2*edgeitems:
                    slices.append((slice(0, edgeitems), slice(-edgeitems, None)))
                else:
                    slices.append((slice(None), ))
            x_flat = np.hstack([x[use_slices].flatten() for use_slices in
                                itertools.product(*slices)])
        else:
            x_flat = np.array(x, copy=False).flatten()
        floatreps = np.tile(np.abs(x_flat), (len(matching), 1)).T / matching_values
        # ignore zeros, they are well represented in any unit
        floatreps[floatreps == 0] = np.nan
        if np.all(np.isnan(floatreps)):
            return matching[1.0]  # all zeros, use the base unit

        deviations = np.nansum((np.log10(floatreps) - 1)**2, axis=0)
        return matching.values()[deviations.argmin()]


def register_new_unit(u):
    """Register a new unit for automatic displaying of quantities

    Parameters
    ----------
    u : `Unit`
        The unit that should be registered.

    Examples
    --------
    >>> from brian2 import *
    >>> 2.0*farad/metre**2
    2. * metre ** -4 * kilogram ** -1 * second ** 4 * amp ** 2
    >>> register_new_unit(pfarad / mmetre**2)
    >>> 2.0*farad/metre**2
    2000000. * pfarad / mmetre ** 2
    """
    user_unit_register.add(u)


#: `UnitRegistry` containing all the standard units (metre, kilogram, um2...)
standard_unit_register = UnitRegistry()
#: `UnitRegistry` containing additional units (newton*metre, farad / metre, ...) 
additional_unit_register = UnitRegistry()
#: `UnitRegistry` containing all units defined by the user
user_unit_register = UnitRegistry()


def get_unit(d):
    '''
    Find an unscaled unit (e.g. `volt` but not `mvolt`) for a `Dimension`.

    Parameters
    ----------
    d : `Dimension`
        The dimension to find a unit for.

    Returns
    -------
    u : `Unit`
        A registered unscaled `Unit` for the dimensions ``d``, or a new `Unit`
        if no unit was found.
    '''
    for unit_register in [standard_unit_register,
                          user_unit_register,
                          additional_unit_register]:
        if 1.0 in unit_register.units_for_dimensions[d]:
            return unit_register.units_for_dimensions[d][1.0]
    return Unit(1.0, dim=d)


def get_unit_for_display(d):
    '''
    Return a string representation of an appropriate unscaled unit or ``'1'``
    for a dimensionless quantity.

    Parameters
    ----------
    d : `Dimension`
        The dimension to find a unit for.

    Returns
    -------
    s : str
        A string representation of the respective unit or the string ``'1'``.
    '''
    if d is 1 or d is DIMENSIONLESS:
        return '1'
    else:
        return repr(get_unit(d))

#### DECORATORS

def check_units(**au):
    """Decorator to check units of arguments passed to a function

    Examples
    --------
    >>> from brian2.units import *
    >>> @check_units(I=amp, R=ohm, wibble=metre, result=volt)
    ... def getvoltage(I, R, **k):
    ...     return I*R

    You don't have to check the units of every variable in the function, and
    you can define what the units should be for variables that aren't
    explicitly named in the definition of the function. For example, the code
    above checks that the variable wibble should be a length, so writing

    >>> getvoltage(1*amp, 1*ohm, wibble=1)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    DimensionMismatchError: Function "getvoltage" variable "wibble" has wrong dimensions, dimensions were (1) (m)

    fails, but

    >>> getvoltage(1*amp, 1*ohm, wibble=1*metre)
    1. * volt

    passes. String arguments or ``None`` are not checked
    
    >>> getvoltage(1*amp, 1*ohm, wibble='hello')
    1. * volt
    
    By using the special name ``result``, you can check the return value of the
    function.

    You can also use ``1`` or ``bool`` as a special value to check for a
    unitless number or a boolean value, respectively:
    >>> @check_units(value=1, absolute=bool, result=bool)
    ... def is_high(value, absolute=False):
    ...     if absolute:
    ...         return abs(value) >= 5
    ...     else:
    ...         return value >= 5

    This will then again raise an error if the argument if not of the expected
    type:
    >>> is_high(7)
    True
    >>> is_high(-7, True)
    True
    >>> is_high(3, 4)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    TypeError: Function "is_high" expected a boolean value for argument "absolute" but got 4.

    Raises
    ------

    DimensionMismatchError
        In case the input arguments or the return value do not have the
        expected dimensions.
    TypeError
        If an input argument or return value was expected to be a boolean but
        is not.

    Notes
    -----
    This decorator will destroy the signature of the original function, and
    replace it with the signature ``(*args, **kwds)``. Other decorators will
    do the same thing, and this decorator critically needs to know the signature
    of the function it is acting on, so it is important that it is the first
    decorator to act on a function. It cannot be used in combination with
    another decorator that also needs to know the signature of the function.

    Note that the ``bool`` type is "strict", i.e. it expects a proper
    boolean value and does not accept 0 or 1. This is not the case the other
    way round, declaring an argument or return value as "1" *does* allow for a
    ``True`` or ``False`` value.
    """
    def do_check_units(f):
        def new_f(*args, **kwds):
            newkeyset = kwds.copy()
            arg_names = f.func_code.co_varnames[0:f.func_code.co_argcount]
            for (n, v) in zip(arg_names, args[0:f.func_code.co_argcount]):
                if (not isinstance(v, (Quantity, basestring, bool))
                        and v is not None
                        and n in au):
                    try:
                        # allow e.g. to pass a Python list of values
                        v = Quantity(v)
                    except TypeError:
                        if have_same_dimensions(au[n], 1):
                            raise TypeError(('Argument %s is not a unitless '
                                             'value/array.') % n)
                        else:
                            raise TypeError(('Argument %s is not a quantity, '
                                             'expected a quantity with dimensions '
                                             '%s') % (n, au[n]))
                newkeyset[n] = v

            for k in newkeyset.iterkeys():
                # string variables are allowed to pass, the presumption is they
                # name another variable. None is also allowed, useful for
                # default parameters
                if (k in au.keys() and not isinstance(newkeyset[k], str) and
                                       not newkeyset[k] is None):

                    if au[k] == bool:
                        if not isinstance(newkeyset[k], bool):
                            error_message = ('Function "{f.__name__}" '
                                             'expected a boolean value '
                                             'for argument "{k}" but got '
                                             '{value}').format(f=f, k=k,
                                                               value=newkeyset[k])
                            raise TypeError(error_message)
                    elif not have_same_dimensions(newkeyset[k], au[k]):
                        error_message = ('Function "{f.__name__}" '
                                         'expected a quantitity with unit '
                                         '{unit} for argument "{k}" but got '
                                         '{value}').format(f=f, k=k,
                                                           unit=repr(au[k]),
                                                           value=newkeyset[k])
                        raise DimensionMismatchError(error_message,
                                                     get_dimensions(newkeyset[k]))

            result = f(*args, **kwds)
            if 'result' in au:
                if au['result'] == bool:
                    if not isinstance(result, bool):
                        error_message = ('The return value of function '
                                         '"{f.__name__}" was expected to be '
                                         'a boolean value, but was of type '
                                         '{type}').format(f=f,
                                                           type=type(result))
                        raise TypeError(error_message)
                elif not have_same_dimensions(result, au['result']):
                    error_message = ('The return value of function '
                                     '"{f.__name__}" was expected to have '
                                     'unit {unit} but was '
                                     '{value}').format(f=f,
                                                       unit=repr(au['result']),
                                                       value=result)
                    raise DimensionMismatchError(error_message,
                                                 get_dimensions(result))
            return result
        new_f._orig_func = f
        new_f.__doc__ = f.__doc__
        new_f.__name__ = f.__name__
        # store the information in the function, necessary when using the
        # function in expressions or equations
        arg_units = []
        all_specified = True
        if hasattr(f, '_orig_arg_names'):
            arg_names = f._orig_arg_names
        else:
            arg_names = f.func_code.co_varnames[:f.func_code.co_argcount]
        for name in arg_names:
            unit = au.get(name, None)
            if unit is None:
                all_specified = False
                break
            else:
                arg_units.append(unit)
        if all_specified:
            new_f._arg_units = arg_units
        else:
            new_f._arg_units = None
        return_unit = au.get('result', None)
        if return_unit is None:
            new_f._return_unit = None
        else:
            new_f._return_unit = return_unit
        if return_unit == bool:
            new_f._returns_bool = True
        else:
            new_f._returns_bool = False
        new_f._orig_arg_names = arg_names

        # copy any annotation attributes
        if hasattr(f, '_annotation_attributes'):
            for attrname in f._annotation_attributes:
                setattr(new_f, attrname, getattr(f, attrname))
        new_f._annotation_attributes = getattr(f, '_annotation_attributes', [])+['_arg_units',
                                                                                 '_return_unit',
                                                                                 '_orig_func',
                                                                                 '_returns_bool']
        return new_f
    return do_check_units
