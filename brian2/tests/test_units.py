import itertools
import warnings
import pickle

from nose.plugins.attrib import attr
import numpy as np
from numpy.testing import assert_raises, assert_equal

import brian2
from brian2.core.preferences import prefs
from brian2.units.fundamentalunits import (UFUNCS_DIMENSIONLESS,
                                           UFUNCS_DIMENSIONLESS_TWOARGS,
                                           UFUNCS_INTEGERS,
                                           UFUNCS_LOGICAL,
                                           Quantity,
                                           Unit,
                                           have_same_dimensions,
                                           get_dimensions,
                                           is_scalar_type,
                                           DimensionMismatchError,
                                           check_units,
                                           in_unit,
                                           get_unit,
                                           get_or_create_dimension,
                                           DIMENSIONLESS,
                                           fail_for_dimension_mismatch)
from brian2.units.allunits import *
from brian2.units.stdunits import ms, mV, kHz, nS, cm, Hz, mM
from brian2.tests.utils import assert_allclose


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


def assert_quantity(q, values, unit):
    assert isinstance(q, Quantity) or (have_same_dimensions(unit, 1) and
                                       (values.shape == () or
                                        isinstance(q, np.ndarray))), q
    assert_allclose(np.asarray(q), values)
    assert have_same_dimensions(q, unit), ('Dimension mismatch: (%s) (%s)' %
                                           (get_dimensions(q),
                                            get_dimensions(unit)))


@attr('codegen-independent')
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
    q = [0.5, 1] * second
    assert_quantity(q, np.array([0.5, 1]), second)

    # dimensionless quantities
    q = Quantity([1, 2, 3])
    assert_quantity(q, np.array([1, 2, 3]), Unit(1))
    q = Quantity(np.array([1, 2, 3]))
    assert_quantity(q, np.array([1, 2, 3]), Unit(1))
    q = Quantity([])
    assert_quantity(q, np.array([]), Unit(1))

    # copying/referencing a quantity
    q1 = Quantity.with_dimensions(np.array([0.5, 1]), second=1)
    q2 = Quantity(q1) # no copy
    assert_quantity(q2, np.asarray(q1), q1)
    q2[0] = 3 * second
    assert_equal(q1[0], 3*second)

    q1 = Quantity.with_dimensions(np.array([0.5, 1]), second=1)
    q2 = Quantity(q1, copy=True) # copy
    assert_quantity(q2, np.asarray(q1), q1)
    q2[0] = 3 * second
    assert_equal(q1[0], 0.5*second)

    # Illegal constructor calls
    assert_raises(TypeError, lambda: Quantity([500 * ms, 1]))
    assert_raises(TypeError, lambda: Quantity(['some', 'nonsense']))
    assert_raises(DimensionMismatchError, lambda: Quantity([500 * ms,
                                                            1 * volt]))


@attr('codegen-independent')
def test_get_dimensions():
    '''
    Test various ways of getting/comparing the dimensions of a quantity.
    '''
    q = 500 * ms
    assert get_dimensions(q) is get_or_create_dimension(q.dimensions._dims)
    assert get_dimensions(q) is q.dimensions
    assert q.has_same_dimensions(3 * second)
    dims = q.dimensions
    assert_equal(dims.get_dimension('time'), 1.)
    assert_equal(dims.get_dimension('length'), 0)
    
    assert get_dimensions(5) is DIMENSIONLESS
    assert get_dimensions(5.0) is DIMENSIONLESS
    assert get_dimensions(np.array(5, dtype=np.int)) is DIMENSIONLESS
    assert get_dimensions(np.array(5.0)) is DIMENSIONLESS
    assert get_dimensions(np.float32(5.0)) is DIMENSIONLESS
    assert get_dimensions(np.float64(5.0)) is DIMENSIONLESS
    assert is_scalar_type(5)
    assert is_scalar_type(5.0)
    assert is_scalar_type(np.array(5, dtype=np.int))
    assert is_scalar_type(np.array(5.0))
    assert is_scalar_type(np.float32(5.0))
    assert is_scalar_type(np.float64(5.0))
    assert_raises(TypeError, lambda: get_dimensions('a string'))
    # wrong number of indices
    assert_raises(TypeError, lambda: get_or_create_dimension([1, 2, 3, 4, 5, 6]))
    # not a sequence
    assert_raises(TypeError, lambda: get_or_create_dimension(42))


@attr('codegen-independent')
def test_display():
    '''
    Test displaying a quantity in different units
    '''
    assert_equal(in_unit(3 * volt, mvolt), '3000. mV')
    assert_equal(in_unit(10 * mV, ohm * amp), '0.01 ohm A')
    assert_raises(DimensionMismatchError, lambda: in_unit(10 * nS, ohm))
    
    # A bit artificial...
    assert_equal(in_unit(10.0, Unit(10.0, scale=1)), '1.0')

@attr('codegen-independent')
def test_scale():
    # Check that unit scaling is implemented correctly
    from brian2.core.namespace import DEFAULT_UNITS
    siprefixes = {"y": 1e-24, "z": 1e-21, "a": 1e-18, "f": 1e-15, "p": 1e-12,
                  "n": 1e-9, "u": 1e-6, "m": 1e-3, "": 1.0, "k": 1e3,
                  "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15, "E": 1e18,
                  "Z": 1e21, "Y": 1e24}
    for prefix in siprefixes:
        if prefix in ['c', 'd', 'da', 'h']:
            continue
        scaled_unit = DEFAULT_UNITS[prefix + 'meter']
        assert_allclose(float(scaled_unit), siprefixes[prefix])
        assert_allclose(5*scaled_unit/meter, 5*siprefixes[prefix])
        scaled_unit = DEFAULT_UNITS[prefix + 'meter2']
        assert_allclose(float(scaled_unit), siprefixes[prefix]**2)
        assert_allclose(5 * scaled_unit / meter2, 5 * siprefixes[prefix] ** 2)
        scaled_unit = DEFAULT_UNITS[prefix + 'meter3']
        assert_allclose(float(scaled_unit), siprefixes[prefix]**3)
        assert_allclose(5 * scaled_unit / meter3, 5 * siprefixes[prefix] ** 3)
        # liter, gram, and molar are special, they are not base units with a
        # value of one, even though they do not have any prefix
        for unit, factor in [('liter', 1e-3),
                             ('litre', 1e-3),
                             ('gram', 1e-3),
                             ('gramme', 1e-3),
                             ('molar', 1e3)]:
            base_unit = DEFAULT_UNITS[unit]
            scaled_unit = DEFAULT_UNITS[prefix + unit]
            assert_allclose(float(scaled_unit), siprefixes[prefix]*factor)
            assert_allclose(5 * scaled_unit / base_unit, 5 * siprefixes[prefix])


@attr('codegen-independent')
def test_pickling():
    '''
    Test pickling of units.
    '''
    for q in [500 * mV, 500 * mV/mV, np.arange(10) * mV,
              np.arange(12).reshape(4, 3) * mV/ms]:
        pickled = pickle.dumps(q)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, type(q))
        assert have_same_dimensions(unpickled, q)
        assert_equal(unpickled, q)


@attr('codegen-independent')
def test_str_repr():
    '''
    Test that str representations do not raise any errors and that repr
    fullfills eval(repr(x)) == x.
    '''
    from numpy import array # necessary for evaluating repr    
    
    units_which_should_exist = [metre, meter, kilogram, kilogramme, second, amp, kelvin, mole, candle,
                                radian, steradian, hertz, newton, pascal, joule, watt,
                                coulomb, volt, farad, ohm, siemens, weber, tesla, henry,
                                lumen, lux, becquerel, gray, sievert, katal,
                                gram, gramme, molar, liter, litre]
    
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
                     np.ones(3) * nS / cm**2,
                     Unit(1, dim=get_or_create_dimension(length=5, time=2)),
                     8000*umetre**3, [0.0001, 10000] * umetre**3,
                     1/metre, 1/(coulomb*metre**2), Unit(1)/second,
                     3.*mM, 5*mole/liter, 7*liter/meter3]
    
    unitless = [second/second, 5 * second/second, Unit(1)]
    
    for u in itertools.chain(units_which_should_exist, some_scaled_units,
                              powered_units, complex_units, unitless):
        assert(len(str(u)) > 0)
        assert_allclose(eval(repr(u)), u)

    # test the `DIMENSIONLESS` object
    assert str(DIMENSIONLESS) == '1'
    assert repr(DIMENSIONLESS) == 'Dimension()'
    
    # test DimensionMismatchError (only that it works without raising an error
    for error in [DimensionMismatchError('A description'),
                  DimensionMismatchError('A description', DIMENSIONLESS),
                  DimensionMismatchError('A description', DIMENSIONLESS,
                                         second.dim)]:
        assert len(str(error))
        assert len(repr(error))


@attr('codegen-independent')
def test_slicing():
    # Slicing and indexing, setting items
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


@attr('codegen-independent')
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


@attr('codegen-independent')
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

        # using unsupported objects should fail
        assert_raises(TypeError, lambda: q / 'string')
        assert_raises(TypeError, lambda: 'string' / q)
        assert_raises(TypeError, lambda: 'string' * q)
        assert_raises(TypeError, lambda: q * 'string')


@attr('codegen-independent')
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
        
        # using unsupported objects should fail
        assert_raises(TypeError, lambda: q + 'string')
        assert_raises(TypeError, lambda: 'string' + q)
        assert_raises(TypeError, lambda: q - 'string')
        assert_raises(TypeError, lambda: 'string' - q)

@attr('codegen-independent')
def test_unary_operations():
    from operator import neg, pos

    for op in [neg, pos]:
        for x in [2, np.array([2]), np.array([1, 2])]:
            assert_quantity(op(x*kilogram), op(x), kilogram)


@attr('codegen-independent')
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

    # Check that comparisons with inf/-inf always work
    values = [2 * kilogram/kilogram,
              2 * kilogram,
              np.array([2]) * kilogram,
              np.array([1, 2]) * kilogram]
    for value in values:
        assert np.all(value < np.inf)
        assert np.all(np.inf > value)
        assert np.all(value <= np.inf)
        assert np.all(np.inf >= value)
        assert np.all(value != np.inf)
        assert np.all(np.inf != value)
        assert np.all(value >= -np.inf)
        assert np.all(-np.inf <= value)
        assert np.all(value > -np.inf)
        assert np.all(-np.inf < value)


@attr('codegen-independent')
def test_power():
    '''
    Test raising quantities to a power.
    '''
    values = [2 * kilogram, np.array([2]) * kilogram,
              np.array([1, 2]) * kilogram]
    for value in values:
        assert_quantity(value ** 3, np.asarray(value) ** 3, kilogram ** 3)
        # Test raising to a dimensionless quantity
        assert_quantity(value ** (3 * volt/volt), np.asarray(value) ** 3, kilogram ** 3)    
        assert_raises(DimensionMismatchError, lambda: value ** (2 * volt))
        assert_raises(TypeError, lambda: value ** np.array([2, 3]))


@attr('codegen-independent')
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
    assert_raises(TypeError, lambda: illegal_pow(np.arange(10)))
    
    # inplace operations with unsupported objects should fail
    for inplace_op in [q.__iadd__, q.__isub__, q.__imul__,
                       q.__idiv__, q.__itruediv__, q.__ifloordiv__,
                       q.__imod__, q.__ipow__]:
        try:
            result = inplace_op('string')
            # if it doesn't fail with an error, it should return NotImplemented
            assert result == NotImplemented
        except TypeError:
            pass  # raised on numpy >= 0.10
    
    # make sure that inplace operations do not work on units/dimensions at all
    for inplace_op in [volt.__iadd__, volt.__isub__, volt.__imul__,
                       volt.__idiv__, volt.__itruediv__, volt.__ifloordiv__,
                       volt.__imod__, volt.__ipow__]:
        assert_raises(TypeError, lambda: inplace_op(volt))
    for inplace_op in [volt.dimensions.__imul__, volt.dimensions.__idiv__,
                       volt.dimensions.__itruediv__,
                       volt.dimensions.__ipow__]:
        assert_raises(TypeError, lambda: inplace_op(volt.dimensions))


@attr('codegen-independent')
def test_unit_discarding_functions():
    '''
    Test functions that discard units.
    '''
    from brian2.units.unitsafefunctions import zeros_like, ones_like
    
    values = [3 * mV, np.array([1, 2]) * mV, np.arange(12).reshape(3, 4) * mV]
    for value in values:
        assert_equal(np.sign(value), np.sign(np.asarray(value)))
        assert_equal(zeros_like(value), np.zeros_like(np.asarray(value)))
        assert_equal(ones_like(value), np.ones_like(np.asarray(value)))
        assert_equal(np.nonzero(value), np.nonzero(np.asarray(value)))


@attr('codegen-independent')
def test_unitsafe_functions():
    '''
    Test the unitsafe functions wrapping their numpy counterparts.
    '''
    from brian2.units.unitsafefunctions import (sin, sinh, arcsin, arcsinh,
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


@attr('codegen-independent')
def test_special_case_numpy_functions():
    '''
    Test a couple of functions/methods that need special treatment.
    '''
    from brian2.units.unitsafefunctions import ravel, diagonal, trace, dot, where
    
    quadratic_matrix = np.reshape(np.arange(9), (3, 3)) * mV

    # Temporarily suppress warnings related to the matplotlib 1.3 bug
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
    
    assert_equal(np.asarray(quadratic_matrix.prod()),
                 np.asarray(quadratic_matrix).prod())
    assert_equal(np.asarray(quadratic_matrix.prod(axis=0)),
                 np.asarray(quadratic_matrix).prod(axis=0))
        
    # Check for correct units
    if use_matplotlib_units_fix:
        assert have_same_dimensions(1, ravel(quadratic_matrix))
    else:
        assert have_same_dimensions(quadratic_matrix, ravel(quadratic_matrix))
    assert have_same_dimensions(quadratic_matrix, trace(quadratic_matrix))
    assert have_same_dimensions(quadratic_matrix, diagonal(quadratic_matrix))
    assert have_same_dimensions(quadratic_matrix[0] ** 2,
                                dot(quadratic_matrix, quadratic_matrix))
    assert have_same_dimensions(quadratic_matrix.prod(axis=0),
                                quadratic_matrix[0] ** quadratic_matrix.shape[0])
    
    # check the where function
    # pure numpy array
    cond = [True, False, False]
    ar1 = np.array([1, 2, 3])
    ar2 = np.array([4, 5, 6])
    assert_equal(np.where(cond), where(cond))
    assert_equal(np.where(cond, ar1, ar2), where(cond, ar1, ar2))
    
    # dimensionless quantity
    assert_equal(np.where(cond, ar1, ar2),
                 np.asarray(where(cond, ar1 * mV/mV, ar2 * mV/mV)))
    
    # quantity with dimensions
    ar1 = ar1 * mV
    ar2 = ar2 * mV
    assert_equal(np.where(cond, np.asarray(ar1), np.asarray(ar2)),
                 np.asarray(where(cond, ar1, ar2)))    
    
    # Check some error cases
    assert_raises(ValueError, lambda: where(cond, ar1))
    assert_raises(TypeError, lambda: where(cond, ar1, ar1, ar2))
    assert_raises(DimensionMismatchError, lambda: where(cond, ar1, ar1 / ms))

    # Check setasflat (for numpy < 1.7)
    if hasattr(Quantity, 'setasflat'):
        a = np.arange(10) * mV
        b = np.ones(10).reshape(5, 2) * volt
        c = np.ones(10).reshape(5, 2) * second
        assert_raises(DimensionMismatchError, lambda: a.setasflat(c))
        a.setasflat(b)
        assert_equal(a.flatten(), b.flatten())

    # Check cumprod
    a = np.arange(1, 10) * mV/mV
    assert_equal(a.cumprod(), np.asarray(a).cumprod())
    assert_raises(TypeError, lambda: (np.arange(1, 5)*mV).cumprod())


# Functions that should not change units
@attr('codegen-independent')
def test_numpy_functions_same_dimensions():
    values = [np.array([1, 2]), np.ones((3, 3))]
    units = [volt, second, siemens, mV, kHz]

    # numpy functions
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
    
    # Python builtins should work on one-dimensional arrays
    value = np.arange(5)
    builtins = [abs, max, min, sum]
    for unit in units:
        q_ar = value * unit
        for func in builtins:
            test_ar = func(q_ar)
            if not get_dimensions(test_ar) is q_ar.dim:
                raise AssertionError(('%s failed on %s -- dim was %s, is now '
                                      '%s') % (func.__name__, repr(q_ar),
                                               q_ar.dim,
                                               get_dimensions(test_ar)))


@attr('codegen-independent')
def test_numpy_functions_indices():
    '''
    Check numpy functions that return indices.
    '''
    values = [np.array([-4, 3, -2, 1, 0]), np.ones((3, 3)), np.array([17])]
    units = [volt, second, siemens, mV, kHz]

    # numpy functions
    keep_dim_funcs = [np.argmin, np.argmax, np.argsort, np.nonzero]

    for value, unit in itertools.product(values, units):
        q_ar = value * unit
        for func in keep_dim_funcs:
            test_ar = func(q_ar)
            # Compare it to the result on the same value without units
            comparison_ar = func(value)
            assert_equal(test_ar, comparison_ar, (('function %s returned an '
                                                   'incorrect result when '
                                                   'used on '
                                                   'quantities ') %
                                                  func.__name__))


@attr('codegen-independent')
def test_numpy_functions_dimensionless():
    '''
    Test that numpy functions that should work on dimensionless quantities only
    work dimensionless arrays and return the correct result.
    '''
    unitless_values = [3, np.array([-4, 3, -1, 2]),
                       np.ones((3, 3))]
    unit_values = [3 * mV, np.array([-4, 3, -1, 2]) * mV,
                       np.ones((3, 3)) * mV]
    with warnings.catch_warnings():
        # ignore division by 0 warnings
        warnings.simplefilter("ignore", RuntimeWarning)    
        for value in unitless_values:
            for ufunc in UFUNCS_DIMENSIONLESS:
                result_unitless = eval('np.%s(value)' % ufunc)
                result_array = eval('np.%s(np.array(value))' % ufunc)
                assert isinstance(result_unitless, (np.ndarray, np.number)) and not isinstance(result_unitless, Quantity)
                assert_equal(result_unitless, result_array)
            for ufunc in UFUNCS_DIMENSIONLESS_TWOARGS:
                result_unitless = eval('np.%s(value, value)' % ufunc)
                result_array = eval('np.%s(np.array(value), np.array(value))' % ufunc)
                assert isinstance(result_unitless, (np.ndarray, np.number)) and not isinstance(result_unitless, Quantity)
                assert_equal(result_unitless, result_array)
        
        for value, unitless_value in zip(unit_values, unitless_values):
            for ufunc in UFUNCS_DIMENSIONLESS:
                assert_raises(DimensionMismatchError,
                              lambda: eval('np.%s(value)' % ufunc,
                                           globals(), {'value': value}))
            for ufunc in UFUNCS_DIMENSIONLESS_TWOARGS:
                assert_raises(DimensionMismatchError,
                              lambda: eval('np.%s(value1, value2)' % ufunc,
                                           globals(), {'value1': value,
                                                       'value2': unitless_value}))
                assert_raises(DimensionMismatchError,
                              lambda: eval('np.%s(value2, value1)' % ufunc,
                                           globals(), {'value1': value,
                                                       'value2': unitless_value}))
                assert_raises(DimensionMismatchError,
                              lambda: eval('np.%s(value, value)' % ufunc,
                                           globals(), {'value': value}))


@attr('codegen-independent')
def test_numpy_functions_change_dimensions():
    '''
    Test some numpy functions that change the dimensions of the quantity.
    '''
    unit_values = [np.array([1, 2]) * mV,
                   np.ones((3, 3)) * 2 * mV]
    for value in unit_values:
        assert_quantity(np.var(value), np.var(np.array(value)), volt ** 2)
        assert_quantity(np.square(value), np.square(np.array(value)),
                        volt ** 2)        
        assert_quantity(np.sqrt(value), np.sqrt(np.array(value)), volt ** 0.5)
        assert_quantity(np.reciprocal(value), np.reciprocal(np.array(value)),
                        1.0 / volt)


@attr('codegen-independent')
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


@attr('codegen-independent')
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
                # assert that comparing to a string results in "NotImplemented" or an error
                try:
                    result = eval('np.%s(value1, "a string")' % ufunc)
                    assert result == NotImplemented
                except TypeError:
                    pass  # raised on numpy >= 0.10
                try:
                    result = eval('np.%s("a string", value1)' % ufunc)
                    assert result == NotImplemented
                except TypeError:
                    pass  # raised on numpy >= 0.10
            assert not isinstance(result_units, Quantity)
            assert_equal(result_units, result_array)


@attr('codegen-independent')
def test_arange_linspace():
    # For dimensionless values, the unit-safe functions should give the same results
    assert_equal(brian2.arange(5), np.arange(5))
    assert_equal(brian2.arange(1, 5), np.arange(1, 5))
    assert_equal(brian2.arange(10, step=2), np.arange(10, step=2))
    assert_equal(brian2.arange(0, 5, 0.5), np.arange(0, 5, 0.5))
    assert_equal(brian2.linspace(0, 1), np.linspace(0, 1))
    assert_equal(brian2.linspace(0, 1, 10), np.linspace(0, 1, 10))

    # Make sure units are checked
    assert_raises(DimensionMismatchError, lambda: brian2.arange(1*mV, 5))
    assert_raises(DimensionMismatchError, lambda: brian2.arange(1*mV, 5*mV))
    assert_raises(DimensionMismatchError, lambda: brian2.arange(1, 5*mV))
    assert_raises(DimensionMismatchError, lambda: brian2.arange(1*mV, 5*ms))
    assert_raises(DimensionMismatchError, lambda: brian2.arange(1*mV, 5*mV, step=1*ms))
    assert_raises(DimensionMismatchError, lambda: brian2.arange(1*ms, 5*mV))

    # Check correct functioning with units
    assert_quantity(brian2.arange(5*mV, step=1*mV), float(mV)*np.arange(5, step=1), mV)
    assert_quantity(brian2.arange(1*mV, 5*mV, 1*mV), float(mV)*np.arange(1, 5, 1), mV)
    assert_quantity(brian2.linspace(1*mV, 2*mV), float(mV)*np.linspace(1, 2), mV)

    # Check errors for arange with incorrect numbers of arguments/duplicate arguments
    assert_raises(TypeError, lambda: brian2.arange())
    assert_raises(TypeError, lambda: brian2.arange(0, 5, 1, 0))
    assert_raises(TypeError, lambda: brian2.arange(0, stop=1))
    assert_raises(TypeError, lambda: brian2.arange(0, 5, stop=1))
    assert_raises(TypeError, lambda: brian2.arange(0, 5, start=1))
    assert_raises(TypeError, lambda: brian2.arange(0, 5, 1, start=1))
    assert_raises(TypeError, lambda: brian2.arange(0, 5, 1, stop=2))
    assert_raises(TypeError, lambda: brian2.arange(0, 5, 1, step=2))


@attr('codegen-independent')
def test_list():
    '''
    Test converting to and from a list.
    '''
    values = [3 * mV, np.array([1, 2]) * mV,
              np.arange(12).reshape(4, 3) * mV]
    for value in values:
        l = value.tolist()
        from_list = Quantity(l)
        assert have_same_dimensions(from_list, value)
        assert_equal(from_list, value)


@attr('codegen-independent')
def test_check_units():
    '''
    Test the check_units decorator
    '''
    @check_units(v=volt)
    def a_function(v, x):
        '''
        v has to have units of volt, x can have any (or no) unit.
        '''
        pass
    
    #Try correct units
    a_function(3 * mV, 5 * second)
    a_function(5 * volt, 'something')
    a_function([1, 2, 3]*volt, None)
    # lists that can be converted should also work
    a_function([1*volt, 2*volt, 3*volt], None)
    # Strings and None are also allowed to pass
    a_function('a string', None)
    a_function(None, None)
    
    # Try incorrect units
    assert_raises(DimensionMismatchError, lambda: a_function(5 * second, None))
    assert_raises(DimensionMismatchError, lambda: a_function(5, None))
    assert_raises(TypeError, lambda: a_function(object(), None))
    assert_raises(TypeError, lambda: a_function([1, 2*volt, 3], None))

    @check_units(result=second)
    def b_function(return_second):
        '''
        Return a value in seconds if return_second is True, otherwise return
        a value in volt.
        '''
        if return_second:
            return 5 * second
        else:
            return 3 * volt
    
    # Should work (returns second)
    b_function(True)
    # Should fail (returns volt)
    assert_raises(DimensionMismatchError, lambda: b_function(False))

    @check_units(a=bool, b=1, result=bool)
    def c_function(a, b):
        if a:
            return b > 0
        else:
            return b

    assert c_function(True, 1)
    assert not c_function(True, -1)
    assert_raises(TypeError, c_function, 1, 1)
    assert_raises(TypeError, c_function, 1*mV, 1)
    assert_raises(TypeError, c_function, False, 1)


@attr('codegen-independent')
def test_get_unit():
    '''
    Test get_unit
    '''
    values = [(volt.dim, volt),
              (mV.dim, volt),
              ((amp/metre**2).dim, amp/metre**2)]
    for dim, expected_unit in values:
        unit = get_unit(dim)
        assert isinstance(unit, Unit)
        assert unit == expected_unit
        assert float(unit) == 1.


@attr('codegen-independent')
def test_get_best_unit():
    # get_best_unit should not check all values for long arrays, since it is
    # a function used for display purposes only. Instead, only the first and
    # last few values should matter (see github issue #966)
    long_ar = np.ones(10000)*siemens
    long_ar[:10] = 1*nS
    long_ar[-10:] = 2*nS
    values = [(np.arange(10)*mV, mV),
              ([0.001, 0.002, 0.003]*second, ms),
              (long_ar, nS)]
    for ar, expected_unit in values:
        assert ar.get_best_unit() is expected_unit
        assert str(expected_unit) in ar.in_best_unit()


@attr('codegen-independent')
def test_switching_off_unit_checks():
    '''
    Check switching off unit checks (used for external functions).
    '''
    import brian2.units.fundamentalunits as fundamentalunits
    x = 3 * second
    y = 5 * volt    
    assert_raises(DimensionMismatchError, lambda: x + y)
    fundamentalunits.unit_checking = False
    # Now it should work
    assert np.asarray(x + y) == np.array(8)    
    assert have_same_dimensions(x, y)
    assert x.has_same_dimensions(y)
    fundamentalunits.unit_checking = True


@attr('codegen-independent')
def test_fail_for_dimension_mismatch():
    '''
    Test the fail_for_dimension_mismatch function.
    '''
    # examples that should not raise an error
    dim1, dim2 = fail_for_dimension_mismatch(3)
    assert dim1 is DIMENSIONLESS
    assert dim2 is DIMENSIONLESS
    dim1, dim2 = fail_for_dimension_mismatch(3 * volt/volt)
    assert dim1 is DIMENSIONLESS
    assert dim2 is DIMENSIONLESS
    dim1, dim2 = fail_for_dimension_mismatch(3 * volt/volt, 7)
    assert dim1 is DIMENSIONLESS
    assert dim2 is DIMENSIONLESS
    dim1, dim2 = fail_for_dimension_mismatch(3 * volt, 5 * volt)
    assert dim1 is volt.dim
    assert dim2 is volt.dim

    # examples that should raise an error
    assert_raises(DimensionMismatchError, lambda: fail_for_dimension_mismatch(6 * volt))
    assert_raises(DimensionMismatchError, lambda: fail_for_dimension_mismatch(6 * volt, 5 * second))    


@attr('codegen-independent')
def test_deepcopy():
    d = {'x': 1*second}
    from copy import deepcopy
    d_copy = deepcopy(d)
    assert d_copy['x'] == 1*second
    d_copy['x'] += 1*second
    assert d_copy['x'] == 2*second
    assert d['x'] == 1*second


@attr('codegen-independent')
def test_inplace_on_scalars():
    # We want "copy semantics" for in-place operations on scalar quantities
    # in the same way as for Python scalars
    for scalar in [3*mV, 3*mV/mV]:
        scalar_reference = scalar
        scalar_copy = Quantity(scalar, copy=True)
        scalar += scalar_copy
        assert_equal(scalar_copy, scalar_reference)
        scalar *= 1.5
        assert_equal(scalar_copy, scalar_reference)
        scalar /= 2
        assert_equal(scalar_copy, scalar_reference)

        # also check that it worked correctly for the scalar itself
        assert_allclose(scalar, (scalar_copy + scalar_copy)*1.5/2)

    # For arrays, it should use reference semantics
    for vector in [[3]*mV, [3]*mV/mV]:
        vector_reference = vector
        vector_copy = Quantity(vector, copy=True)
        vector += vector_copy
        assert_equal(vector, vector_reference)
        vector *= 1.5
        assert_equal(vector, vector_reference)
        vector /= 2
        assert_equal(vector, vector_reference)

        # also check that it worked correctly for the vector itself
        assert_allclose(vector, (vector_copy + vector_copy)*1.5/2)


def test_units_vs_quantities():
    # Unit objects should stay Unit objects under certain operations
    # (important e.g. in the unit definition of Equations, where only units but
    # not quantities are allowed)
    assert type(meter**2) == Unit
    assert type(meter**-1) == Unit
    assert type(meter**0.5) == Unit
    assert type(meter/second) == Unit
    assert type(amp/meter**2) == Unit
    assert type(1/meter) == Unit
    assert type(1.0/meter) == Unit

    assert type(2/meter) == Quantity
    assert type(2*meter) == Quantity
    assert type(meter + meter) == Quantity
    assert type(meter - meter) == Quantity


@attr('codegen-independent')
def test_all_units_list():
    from brian2.units.allunits import all_units
    assert meter in all_units
    assert volt in all_units
    assert cm in all_units
    assert Hz in all_units
    assert all(isinstance(u, Unit) for u in all_units)

@attr('codegen-independent')
def test_constants():
    import brian2.units.constants as constants
    # Check that the expected names exist and have the correct dimensions
    assert constants.avogadro_constant.dim == (1/mole).dim
    assert constants.boltzmann_constant.dim == (joule/kelvin).dim
    assert constants.electric_constant.dim == (farad/meter).dim
    assert constants.electron_mass.dim == kilogram.dim
    assert constants.elementary_charge.dim == coulomb.dim
    assert constants.faraday_constant.dim == (coulomb/mole).dim
    assert constants.gas_constant.dim == (joule/mole/kelvin).dim
    assert constants.magnetic_constant.dim == (newton/amp2).dim
    assert constants.molar_mass_constant.dim == (kilogram/mole).dim
    assert constants.zero_celsius.dim == kelvin.dim

    # Check the consistency between a few constants
    assert_allclose(constants.gas_constant,
                    constants.avogadro_constant*constants.boltzmann_constant)
    assert_allclose(constants.faraday_constant,
                    constants.avogadro_constant*constants.elementary_charge)


if __name__ == '__main__':
    test_construction()
    test_get_dimensions()
    test_display()
    test_scale()
    test_power()
    test_pickling()
    test_str_repr()
    test_slicing()
    test_setting()
    test_multiplication_division()
    test_addition_subtraction()
    test_unary_operations()
    test_binary_operations()
    test_inplace_operations()
    test_unit_discarding_functions()
    test_unitsafe_functions()
    test_special_case_numpy_functions()
    test_numpy_functions_same_dimensions()
    test_numpy_functions_indices()
    test_numpy_functions_dimensionless()
    test_numpy_functions_change_dimensions()
    test_numpy_functions_typeerror()
    test_numpy_functions_logical()
    test_arange_linspace()
    test_list()
    test_check_units()
    test_get_unit()
    test_get_best_unit()
    test_switching_off_unit_checks()
    test_fail_for_dimension_mismatch()
    test_deepcopy()
    test_inplace_on_scalars()
    test_units_vs_quantities()
    test_all_units_list()
    test_constants()
