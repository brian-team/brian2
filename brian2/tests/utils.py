
import numpy as np
from numpy.testing import assert_allclose as numpy_allclose

from brian2 import prefs
from brian2.units.fundamentalunits import have_same_dimensions


def assert_allclose(actual, desired, rtol=4.5e8, atol=0, **kwds):
    '''
    Thin wrapper around numpy's `~numpy.testing.utils.assert_allclose` function. The tolerance depends on the floating
    point precision as defined by the `core.default_float_dtype` preference.

    Parameters
    ----------
    actual : `numpy.ndarray`
        The results to check.
    desired : `numpy.ndarray`
        The expected results.
    rtol : float, optional
        The relative tolerance. Will be multiplied with the machine epsilon for 64bit float numbers, and adjusted
        to be the same tolerance in logarithmic terms for 32 bit numbers. For example. the default value of 4.5e8 leads
        to a tolerance of 1e-7 for 64 bit floating point numbers, where the machine epsilon is on the order of 1e-16. It
        therefore corresponds to about half of the available decimals in a 64 bit float. For 32 bit floating point
        numbers, where the machine epsilon is on the order of 1e-7, the tolerance will then be about 8e-4.
    atol : float, optional
        The absolute tolerance which will be multiplied with the machine epsilon of the type set as
        `core.default_float_type`.
    '''
    assert have_same_dimensions(actual, desired)
    float_dtype = prefs['core.default_float_dtype']
    reference_eps = np.finfo(np.float64).eps
    eps = np.finfo(float_dtype).eps
    rel_precision = (np.log10(rtol * reference_eps)) / np.log10(reference_eps)
    rtol = 10**(rel_precision * np.log10(eps))
    atol = eps*atol
    numpy_allclose(np.asarray(actual), np.asarray(desired), rtol=rtol, atol=atol, **kwds)
