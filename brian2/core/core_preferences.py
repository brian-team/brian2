'''
Definitions, documentation, default values and validation functions for core
Brian preferences.
'''
from numpy import float64, int32

from brian2.core.preferences import BrianPreference, prefs


def dtype_repr(dtype):
    return dtype.__name__

prefs.register_preferences('core', 'Core Brian preferences',
    default_float_dtype=BrianPreference(
        default=float64,
        docs='''
        Default dtype for all arrays of scalars (state variables, weights, etc.).
        ''',
        representor=dtype_repr,
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
