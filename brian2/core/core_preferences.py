'''
Definitions, documentation, default values and validation functions for core
Brian preferences.
'''
from numpy import float64, int32

from brian2.core.preferences import BrianPreference, prefs


def dtype_repr(dtype):
    return dtype.__name__

def default_float_dtype_validator(dtype):
    return dtype is float64

prefs.register_preferences('core', 'Core Brian preferences',
    default_float_dtype=BrianPreference(
        default=float64,
        docs='''
        Default dtype for all arrays of scalars (state variables, weights, etc.).

        Currently, this is not supported (only float64 can be used).
        ''',
        representor=dtype_repr,
        validator=default_float_dtype_validator,
        ),
    default_integer_dtype=BrianPreference(
        default=int32,
        docs='''
        Default dtype for all arrays of integer scalars.
        ''',
        representor=dtype_repr,
        ),
    outdated_dependency_error=BrianPreference(
        default=True,
        docs='''
        Whether to raise an error for outdated dependencies (``True``) or just
        a warning (``False``).
        '''
        )
    )
