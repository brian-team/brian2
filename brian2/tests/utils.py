import numpy as np
from numpy.testing import assert_allclose as numpy_allclose

from brian2 import prefs
from brian2.units.fundamentalunits import have_same_dimensions


def assert_allclose(actual, desired, rtol=4.5e8, atol=0, **kwds):
    """
    Thin wrapper around numpy's `~numpy.testing.utils.assert_allclose` function. The tolerance depends on the floating
    point precision as defined by the `core.default_float_dtype` preference.

    Parameters
    ----------
    actual : `numpy.ndarray`
        The results to check.
    desired : `numpy.ndarray`
        The expected results.
    rtol : float, optional
        The relative tolerance which will be multiplied with the machine epsilon of the type set as
        `core.default_float_type`.
    atol : float, optional
        The absolute tolerance
    """
    assert have_same_dimensions(actual, desired)
    eps = np.finfo(prefs["core.default_float_dtype"]).eps
    rtol = eps * rtol
    numpy_allclose(
        np.asarray(actual), np.asarray(desired), rtol=rtol, atol=atol, **kwds
    )


def exc_isinstance(exc_info, expected_exception, raise_not_implemented=False):
    """
    Simple helper function as an alternative to calling
    `~.pytest.ExceptionInfo.errisinstance` which will take into account all
    the "causing" exceptions in an exception chain.

    Parameters
    ----------
    exc_info : `pytest.ExceptionInfo` or `Exception`
        The exception info as returned by `pytest.raises`.
    expected_exception : `type`
        The expected exception class
    raise_not_implemented : bool, optional
        Whether to re-raise a `NotImplementedError` – necessary for tests that
        should be skipped with ``@skip_if_not_implemented``. Defaults to
        ``False``.

    Returns
    -------
    correct_exception : bool
        Whether the exception itself or one of the causing exceptions is of the
        expected type.
    """
    if exc_info is None:
        return False
    if hasattr(exc_info, "value"):
        exc_info = exc_info.value

    if isinstance(exc_info, expected_exception):
        return True
    elif raise_not_implemented and isinstance(exc_info, NotImplementedError):
        raise exc_info

    return exc_isinstance(
        exc_info.__cause__,
        expected_exception,
        raise_not_implemented=raise_not_implemented,
    )
