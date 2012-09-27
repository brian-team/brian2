import itertools
import warnings
from exceptions import RuntimeWarning

import numpy as np
from numpy.testing import assert_raises, assert_equal

from brian2.units.fundamentalunits import (UFUNCS_DIMENSIONLESS,
                                           UFUNCS_DIMENSIONLESS_TWOARGS,
                                           UFUNCS_INTEGERS,
                                           UFUNCS_LOGICAL,
                                           Quantity,
                                           have_same_dimensions,
                                           get_dimensions,
                                           DimensionMismatchError)
from brian2.units.allunits import *
from brian2.units.stdunits import ms, mV, kHz, nS, cm


def assert_quantity(q, values, unit):
    assert isinstance(q, Quantity)
    assert np.all(np.asarray(q) == values)
    assert have_same_dimensions(q, unit), ('Dimension mismatch: (%s) (%s)' %
                                           (get_dimensions(q),
                                            get_dimensions(unit)))

# Construct quantities
def test_construction():
    ''' Test the construction of quantity objects '''

    q = 500 * ms
    assert_quantity(q, 0.5, second)
    q = np.float64(500) * ms
    assert_quantity(q, 0.5, second)
    q = np.array(500) * ms
    assert_quantity(q, 0.5, second)
    q = np.array([500, 1000]) * ms
    assert_quantity(q, np.array([0.5, 1]), second)
    q = Quantity(500)
    assert_quantity(q, 500, 1)
    q = Quantity(500, dim=second.dim)
    assert_quantity(q, 500, second)
    q = Quantity([0.5, 1], dim=second.dim)
    assert_quantity(q, np.array([0.5, 1]), second)
    q = Quantity(np.array([0.5, 1]), dim=second.dim)
    assert_quantity(q, np.array([0.5, 1]), second)
    q = Quantity([500 * ms, 1 * second])
    assert_quantity(q, np.array([0.5, 1]), second)
    q = Quantity.with_dimensions(np.array([0.5, 1]), second=1)
    assert_quantity(q, np.array([0.5, 1]), second)

    # Illegal constructor calls
    assert_raises(ValueError, lambda: Quantity([500 * ms, 1]))
    assert_raises(DimensionMismatchError, lambda: Quantity([500 * ms,
                                                            1 * volt]))
    assert_raises(DimensionMismatchError, lambda: Quantity([500 * ms],
                                                           dim=volt.dim))

def test_str_repr():
    '''
    Test that str representations do not raise any errors and that repr
    fullfills eval(repr(x)) == x.
    '''
    from numpy import array # necessary for evaluating repr    
    
    units_which_should_exist = [metre, meter, kilogram, second, amp, kelvin, mole, candle,
                                radian, steradian, hertz, newton, pascal, joule, watt,
                                coulomb, volt, farad, ohm, siemens, weber, tesla, henry,
                                celsius, lumen, lux, becquerel, gray, sievert, katal,
                                gram, gramme]
    
    # scaled versions of all these units should exist (we just check farad as an example)
    some_scaled_units = [Yfarad, Zfarad, Efarad, Pfarad, Tfarad, Gfarad, Mfarad, kfarad,
                         hfarad, dafarad, dfarad, cfarad, mfarad, ufarad, nfarad, pfarad,
                         ffarad, afarad, zfarad, yfarad]
    
    # some powered units
    powered_units = [cmetre2, Yfarad3]
    
    # Combined units
    complex_units = [(kgram * metre2)/(amp * second3),
                     5 * (kgram * metre2)/(amp * second3),
                     metre * second**-1, 10 * metre * second**-1,
                     array([1, 2, 3]) * kmetre / second,
                     np.ones(3) * nS / cm**2]
        
    for u in itertools.chain(units_which_should_exist, some_scaled_units,
                              powered_units, complex_units):
        assert(len(str(u)) > 0)
        assert_equal(eval(repr(u)), u)


# Slicing and indexing, setting items
def test_slicing():
    quantity = np.reshape(np.arange(6), (2, 3)) * mV
    assert_equal(quantity[:], quantity)
    assert_equal(quantity[0], np.asarray(quantity)[0] * volt)
    assert_equal(quantity[0:1], np.asarray(quantity)[0:1] * volt)
    assert_equal(quantity[0, 1], np.asarray(quantity)[0, 1] * volt)
    assert_equal(quantity[0:1, 1:], np.asarray(quantity)[0:1, 1:] * volt)
    bool_matrix = np.array([[True, False, False],
                            [False, False, True]])
    assert_equal(quantity[bool_matrix],
                 np.asarray(quantity)[bool_matrix] * volt)

def test_setting():
    quantity = np.reshape(np.arange(6), (2, 3)) * mV
    quantity[0, 1] = 10 * mV
    assert quantity[0, 1] == 10 * mV
    quantity[:, 1] = 20 * mV
    assert np.all(quantity[:, 1] == 20 * mV)
    quantity[1, :] = np.ones((1, 3)) * volt
    assert np.all(quantity[1, :] == 1 * volt)
    # Setting to zero should work without units as well
    quantity[1, 2] = 0
    assert quantity[1, 2] == 0 * mV
    
    def set_to_value(key, value):
        quantity[key] = value

    assert_raises(DimensionMismatchError, lambda : set_to_value(0, 1))
    assert_raises(DimensionMismatchError, lambda : set_to_value(0, 1 * second))
    assert_raises(DimensionMismatchError, lambda : set_to_value((slice(2), slice(3)),
                                                                np.ones((2, 3))))
    
    # A dimensionless array can be set to values without units
    dimensionless = quantity / mV
    dimensionless[0, 1] = 10
    assert dimensionless[0, 1] == 10
    dimensionless[0, 1] = 30 * volt/volt
    assert dimensionless[0, 1] == 30
    dimensionless[:, 1] = 20
    assert np.all(dimensionless[:, 1] == 20)
    dimensionless[1, :] = np.ones((1, 3))
    assert np.all(dimensionless[1, :] == 1)

    def set_to_value2(key, value):
        dimensionless[key] = value

    assert_raises(DimensionMismatchError, lambda : set_to_value2(0, 1 * second))    
    assert_raises(DimensionMismatchError, lambda : set_to_value2((slice(2), slice(3)),
                                                                np.ones((2, 3)) * volt))
    

# Binary operations
def test_multiplication_division():
    quantities = [3 * mV, np.array([1, 2]) * mV, np.ones((3, 3)) * mV]
    q2 = 5 * second

    for q in quantities:
        # Scalars and array scalars
        assert_quantity(q / 3, np.asarray(q) / 3, volt)
        assert_quantity(3 / q, 3 / np.asarray(q), 1 / volt)
        assert_quantity(q * 3, np.asarray(q) * 3, volt)
        assert_quantity(3 * q, 3 * np.asarray(q), volt)
        assert_quantity(q / np.float64(3), np.asarray(q) / 3, volt)
        assert_quantity(np.float64(3) / q, 3 / np.asarray(q), 1 / volt)
        assert_quantity(q * np.float64(3), np.asarray(q) * 3, volt)
        assert_quantity(np.float64(3) * q, 3 * np.asarray(q), volt)
        assert_quantity(q / np.array(3), np.asarray(q) / 3, volt)
        assert_quantity(np.array(3) / q, 3 / np.asarray(q), 1 / volt)
        assert_quantity(q * np.array(3), np.asarray(q) * 3, volt)
        assert_quantity(np.array(3) * q, 3 * np.asarray(q), volt)

        # (unitless) arrays
        assert_quantity(q / np.array([3]), np.asarray(q) / 3, volt)
        assert_quantity(np.array([3]) / q, 3 / np.asarray(q), 1 / volt)
        assert_quantity(q * np.array([3]), np.asarray(q) * 3, volt)
        assert_quantity(np.array([3]) * q, 3 * np.asarray(q), volt)

        # arrays with units
        assert_quantity(q / q, np.asarray(q) / np.asarray(q), 1)
        assert_quantity(q * q, np.asarray(q) ** 2, volt ** 2)
        assert_quantity(q / q2, np.asarray(q) / np.asarray(q2), volt / second)
        assert_quantity(q2 / q, np.asarray(q2) / np.asarray(q), second / volt)
        assert_quantity(q * q2, np.asarray(q) * np.asarray(q2), volt * second)


def test_addition_subtraction():
    quantities = [3 * mV, np.array([1, 2]) * mV, np.ones((3, 3)) * mV]
    q2 = 5 * volt

    for q in quantities:
        # arrays with units
        assert_quantity(q + q, np.asarray(q) + np.asarray(q), volt)
        assert_quantity(q - q, 0, volt)
        assert_quantity(q + q2, np.asarray(q) + np.asarray(q2), volt)
        assert_quantity(q2 + q, np.asarray(q2) + np.asarray(q), volt)
        assert_quantity(q - q2, np.asarray(q) - np.asarray(q2), volt)
        assert_quantity(q2 - q, np.asarray(q2) - np.asarray(q), volt)
        
        # mismatching units
        assert_raises(DimensionMismatchError, lambda: q + 5 * second)
        assert_raises(DimensionMismatchError, lambda: 5 * second + q)
        assert_raises(DimensionMismatchError, lambda: q - 5 * second)
        assert_raises(DimensionMismatchError, lambda: 5 * second - q)
        
        # scalar        
        assert_raises(DimensionMismatchError, lambda: q + 5)
        assert_raises(DimensionMismatchError, lambda: 5 + q)
        assert_raises(DimensionMismatchError, lambda: q + np.float64(5))
        assert_raises(DimensionMismatchError, lambda: np.float64(5) + q)
        assert_raises(DimensionMismatchError, lambda: q - 5)
        assert_raises(DimensionMismatchError, lambda: 5 - q)
        assert_raises(DimensionMismatchError, lambda: q - np.float64(5))
        assert_raises(DimensionMismatchError, lambda: np.float64(5) - q)
        
        # unitless array
        assert_raises(DimensionMismatchError, lambda: q + np.array([5]))
        assert_raises(DimensionMismatchError, lambda: np.array([5]) + q)
        assert_raises(DimensionMismatchError,
                      lambda: q + np.array([5], dtype=np.float64))
        assert_raises(DimensionMismatchError,
                      lambda: np.array([5], dtype=np.float64) + q)
        assert_raises(DimensionMismatchError, lambda: q - np.array([5]))
        assert_raises(DimensionMismatchError, lambda: np.array([5]) - q)
        assert_raises(DimensionMismatchError,
                      lambda: q - np.array([5], dtype=np.float64))
        assert_raises(DimensionMismatchError,
                      lambda: np.array([5], dtype=np.float64) - q)                        

        # Check that operations with 0 work
        assert_quantity(q + 0, np.asarray(q), volt)
        assert_quantity(0 + q, np.asarray(q), volt)
        assert_quantity(q - 0, np.asarray(q), volt)
        assert_quantity(0 - q, -np.asarray(q), volt)
        assert_quantity(q + np.float64(0), np.asarray(q), volt)
        assert_quantity(np.float64(0) + q, np.asarray(q), volt)
        assert_quantity(q - np.float64(0), np.asarray(q), volt)
        assert_quantity(np.float64(0) - q, -np.asarray(q), volt)

def test_binary_operations():
    ''' Test whether binary operations work when they should and raise
    DimensionMismatchErrors when they should.
    Does not test for the actual result.
    '''
    from operator import add, sub, lt, le, gt, ge, eq, ne

    def assert_operations_work(a, b):
        try:
            # Test python builtins
            tryops = [add, sub, lt, le, gt, ge, eq, ne]
            for op in tryops:
                op(a, b)
                op(b, a)

            # Test equivalent numpy functions
            numpy_funcs = [np.add, np.subtract, np.less, np.less_equal,
                           np.greater, np.greater_equal, np.equal,
                           np.not_equal, np.maximum, np.minimum]
            for numpy_func in numpy_funcs:
                numpy_func(a, b)
                numpy_func(b, a)
        except DimensionMismatchError as ex:
            raise AssertionError('Operation raised unexpected '
                                 'exception: %s' % ex)

    def assert_operations_do_not_work(a, b):
        # Test python builtins
        tryops = [add, sub, lt, le, gt, ge, eq, ne]
        for op in tryops:
            assert_raises(DimensionMismatchError, lambda: op(a, b))
            assert_raises(DimensionMismatchError, lambda: op(b, a))

        # Test equivalent numpy functions
        numpy_funcs = [np.add, np.subtract, np.less, np.less_equal,
                       np.greater, np.greater_equal, np.equal, np.not_equal,
                       np.maximum, np.minimum]
        for numpy_func in numpy_funcs:
            assert_raises(DimensionMismatchError, lambda: numpy_func(a, b))
            assert_raises(DimensionMismatchError, lambda: numpy_func(b, a))

    #
    # Check that consistent units work
    #

    # unit arrays
    a = 1 * kilogram
    for b in [2 * kilogram, np.array([2]) * kilogram,
              np.array([1, 2]) * kilogram]:
        assert_operations_work(a, b)

    # dimensionless units and scalars
    a = 1
    for b in [2 * kilogram / kilogram, np.array([2]) * kilogram / kilogram,
              np.array([1, 2]) * kilogram / kilogram]:
        assert_operations_work(a, b)

    # dimensionless units and unitless arrays
    a = np.array([1])
    for b in [2 * kilogram / kilogram, np.array([2]) * kilogram / kilogram,
              np.array([1, 2]) * kilogram / kilogram]:
        assert_operations_work(a, b)

    #
    # Check that inconsistent units do not work
    #

    # unit arrays
    a = np.array([1]) * second
    for b in [2 * kilogram, np.array([2]) * kilogram,
              np.array([1, 2]) * kilogram]:
        assert_operations_do_not_work(a, b)

    # unitless array
    a = np.array([1])
    for b in [2 * kilogram, np.array([2]) * kilogram,
              np.array([1, 2]) * kilogram]:
        assert_operations_do_not_work(a, b)

    # scalar
    a = 1
    for b in [2 * kilogram, np.array([2]) * kilogram,
              np.array([1, 2]) * kilogram]:
        assert_operations_do_not_work(a, b)

    # TODO: Comparisons to inf/-inf (?)

def test_inplace_operations():
    q = np.arange(10) * volt
    q_orig = q.copy()
    q_id = id(q)
    
    q *= 2
    assert np.all(q == 2 * q_orig) and id(q) == q_id 
    q /= 2
    assert np.all(q == q_orig) and id(q) == q_id
    q += 1 * volt
    assert np.all(q == q_orig + 1 * volt) and id(q) == q_id
    q -= 1 * volt
    assert np.all(q == q_orig) and id(q) == q_id
    q **= 2
    assert np.all(q == q_orig**2) and id(q) == q_id
    q **= 0.5
    assert np.all(q == q_orig) and id(q) == q_id

    def illegal_add(q2):
        q = np.arange(10) * volt
        q += q2
    assert_raises(DimensionMismatchError, lambda: illegal_add(1 * second))
    assert_raises(DimensionMismatchError, lambda: illegal_add(1))

    def illegal_sub(q2):
        q = np.arange(10) * volt
        q -= q2
    assert_raises(DimensionMismatchError, lambda: illegal_add(1 * second))
    assert_raises(DimensionMismatchError, lambda: illegal_add(1))
    
    def illegal_pow(q2):
        q = np.arange(10) * volt
        q **= q2
    assert_raises(DimensionMismatchError, lambda: illegal_pow(1 * volt)) 
    

def test_unitsafe_functions():
    '''
    Test the unitsafe functions wrapping their numpy counterparts.
    '''
    from brian2 import (sin, sinh, arcsin, arcsinh,
                        cos, cosh, arccos, arccosh,
                        tan, tanh, arctan, arctanh,
                        log, exp)
    
    # All functions with their numpy counterparts
    funcs = [(sin, np.sin), (sinh, np.sinh), (arcsin, np.arcsin), (arcsinh, np.arcsinh),
             (cos, np.cos), (cosh, np.cosh), (arccos, np.arccos), (arccosh, np.arccosh),
             (tan, np.tan), (tanh, np.tanh), (arctan, np.arctan), (arctanh, np.arctanh),
             (log, np.log), (exp, np.exp)]
    
    unitless_values = [3 * mV/mV, np.array([1, 2]) * mV/mV,
                       np.ones((3, 3)) * mV/mV]
    numpy_values = [3, np.array([1, 2]),
                       np.ones((3, 3))]
    unit_values = [3 * mV, np.array([1, 2]) * mV,
                       np.ones((3, 3)) * mV]
        
    for func, np_func in funcs:
        #make sure these functions raise errors when run on values with dimensions
        for val in unit_values:
            assert_raises(DimensionMismatchError, lambda : func(val))
        
        # make sure the functions are equivalent to their numpy counterparts
        # when run on unitless values while ignoring warnings about invalid
        # values or divisions by zero        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for val in unitless_values:
                assert_equal(func(val), np_func(val))

            for val in numpy_values:
                assert_equal(func(val), np_func(val))

def test_special_case_numpy_functions():
    '''
    Test a couple of functions that need special treatment because they do
    not automatically call the corresponding method.
    '''
    from brian2 import ravel, diagonal, trace, dot
    
    quadratic_matrix = np.reshape(np.arange(9), (3, 3)) * mV
    # Check that function and method do the same thing
    assert_equal(ravel(quadratic_matrix), quadratic_matrix.ravel())
    # Check that function gives the same result as on unitless arrays
    assert_equal(np.asarray(ravel(quadratic_matrix)),
                 ravel(np.asarray(quadratic_matrix)))
    # Check that the function gives the same results as the original numpy
    # function
    assert_equal(np.ravel(np.asarray(quadratic_matrix)),
                 ravel(np.asarray(quadratic_matrix)))

    # Do the same checks for diagonal, trace and dot
    assert_equal(diagonal(quadratic_matrix), quadratic_matrix.diagonal())
    assert_equal(np.asarray(diagonal(quadratic_matrix)),
                 diagonal(np.asarray(quadratic_matrix)))
    assert_equal(np.diagonal(np.asarray(quadratic_matrix)),
                 diagonal(np.asarray(quadratic_matrix)))

    assert_equal(trace(quadratic_matrix), quadratic_matrix.trace())
    assert_equal(np.asarray(trace(quadratic_matrix)),
                 trace(np.asarray(quadratic_matrix)))
    assert_equal(np.trace(np.asarray(quadratic_matrix)),
                 trace(np.asarray(quadratic_matrix)))

    assert_equal(dot(quadratic_matrix, quadratic_matrix),
                 quadratic_matrix.dot(quadratic_matrix))
    assert_equal(np.asarray(dot(quadratic_matrix, quadratic_matrix)),
                 dot(np.asarray(quadratic_matrix), np.asarray(quadratic_matrix)))
    assert_equal(np.dot(np.asarray(quadratic_matrix), np.asarray(quadratic_matrix)),
                 dot(np.asarray(quadratic_matrix), np.asarray(quadratic_matrix)))
    
    # Check for correct units
    assert have_same_dimensions(quadratic_matrix, ravel(quadratic_matrix))
    assert have_same_dimensions(quadratic_matrix, trace(quadratic_matrix))
    assert have_same_dimensions(quadratic_matrix, diagonal(quadratic_matrix))
    assert have_same_dimensions(quadratic_matrix[0] ** 2,
                                dot(quadratic_matrix, quadratic_matrix))

# Functions that should not change units
def test_numpy_functions_same_dimensions():
    values = [3, np.array([1, 2]), np.ones((3, 3))]
    units = [volt, second, siemens, mV, kHz]

    keep_dim_funcs = [np.abs, np.cumsum, np.max, np.mean, np.min, np.negative,
                      np.ptp, np.round, np.squeeze, np.std, np.sum,
                      np.transpose]

    for value, unit in itertools.product(values, units):
        q_ar = value * unit
        for func in keep_dim_funcs:
            test_ar = func(q_ar)
            if not get_dimensions(test_ar) is q_ar.dim:
                raise AssertionError(('%s failed on %s -- dim was %s, is now '
                                      '%s') % (func.__name__, repr(q_ar),
                                               q_ar.dim,
                                               get_dimensions(test_ar)))

def test_numpy_functions_dimensionless():
    '''
    Test that numpy functions that should work on dimensionless quantities only
    work dimensionless arrays and return the correct result.
    '''
    unitless_values = [3 * mV/mV, np.array([1, 2]) * mV/mV,
                       np.ones((3, 3)) * mV/mV]
    unit_values = [3 * mV, np.array([1, 2]) * mV,
                       np.ones((3, 3)) * mV]
    with warnings.catch_warnings():
        # ignore division by 0 warnings
        warnings.simplefilter("ignore", RuntimeWarning)    
        for value in unitless_values:
            for ufunc in UFUNCS_DIMENSIONLESS:
                result_unitless = eval('np.%s(value)' % ufunc)
                result_array = eval('np.%s(np.array(value))' % ufunc)
                assert result_unitless.is_dimensionless
                assert_equal(np.asarray(result_unitless), result_array)
            for ufunc in UFUNCS_DIMENSIONLESS_TWOARGS:
                result_unitless = eval('np.%s(value, value)' % ufunc)
                result_array = eval('np.%s(np.array(value), np.array(value))' % ufunc)
                assert result_unitless.is_dimensionless
                assert_equal(np.asarray(result_unitless), result_array)
        
        for value in unit_values:
            for ufunc in UFUNCS_DIMENSIONLESS:
                assert_raises(DimensionMismatchError,
                              lambda: eval('np.%s(value)' % ufunc,
                                           globals(), {'value': value}))
            for ufunc in UFUNCS_DIMENSIONLESS_TWOARGS:
                assert_raises(DimensionMismatchError,
                              lambda: eval('np.%s(value, value)' % ufunc,
                                           globals(), {'value': value}))

def test_numpy_functions_change_dimensions():
    '''
    Test some numpy functions that change the dimensions of the quantity.
    '''
    unit_values = [3 * mV, np.array([1, 2]) * mV,
                       np.ones((3, 3)) * 2 * mV]
    for value in unit_values:
        assert_quantity(np.var(value), np.var(np.array(value)), volt ** 2)
        assert_quantity(np.square(value), np.square(np.array(value)),
                        volt ** 2)        
        assert_quantity(np.sqrt(value), np.sqrt(np.array(value)), volt ** 0.5)
        assert_quantity(np.reciprocal(value), np.reciprocal(np.array(value)),
                        1.0 / volt)


def test_numpy_functions_typeerror():
    '''
    Assures that certain numpy functions raise a TypeError when called on
    quantities.
    '''
    unitless_values = [3 * mV/mV, np.array([1, 2]) * mV/mV,
                       np.ones((3, 3)) * mV/mV]
    unit_values = [3 * mV, np.array([1, 2]) * mV,
                       np.ones((3, 3)) * mV]
    for value in unitless_values + unit_values:
        for ufunc in UFUNCS_INTEGERS:
            if ufunc == 'invert': 
                # only takes one argument
                assert_raises(TypeError, lambda: eval('np.%s(value)' % ufunc,
                                                   globals(), {'value': value}))
            else:
                assert_raises(TypeError, lambda: eval('np.%s(value, value)' % ufunc,
                                                   globals(), {'value': value}))

def test_numpy_functions_logical():
    '''
    Assure that logical numpy functions work on all quantities and return
    unitless boolean arrays.
    '''
    unit_values1 = [3 * mV, np.array([1, 2]) * mV, np.ones((3, 3)) * mV]
    unit_values2 = [3 * second, np.array([1, 2]) * second,
                    np.ones((3, 3)) * second]
    for ufunc in UFUNCS_LOGICAL:
        for value1, value2 in zip(unit_values1, unit_values2):
            try:
                # one argument
                result_units = eval('np.%s(value1)' % ufunc)        
                result_array = eval('np.%s(np.array(value1))' % ufunc)
            except ValueError:
                # two arguments
                result_units = eval('np.%s(value1, value2)' % ufunc)        
                result_array = eval('np.%s(np.array(value1), np.array(value2))' % ufunc)
            assert not isinstance(result_units, Quantity)
            assert_equal(result_units, result_array)


if __name__ == '__main__':
    test_construction()
    test_str_repr()
    test_slicing()
    test_setting()
    test_multiplication_division()
    test_addition_subtraction()
    test_binary_operations()
    test_inplace_operations()
    test_unitsafe_functions()
    test_special_case_numpy_functions()    
    test_numpy_functions_same_dimensions()
    test_numpy_functions_dimensionless()
    test_numpy_functions_change_dimensions()
    test_numpy_functions_typeerror()
    test_numpy_functions_logical()
    