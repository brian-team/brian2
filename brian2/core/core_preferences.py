'''
Definitions, documentation, default values and validation functions for core
Brian preferences.
'''
from numpy import float64

from brian2.core.preferences import BrianPreference, brian_prefs

def dtype_repr(dtype):
    return dtype.__name__

brian_prefs.register_preferences('core', 'Core Brian preferences',
    default_scalar_dtype=BrianPreference(
        default=float64,
        docs='''
        Default dtype for all arrays of scalars (state variables, weights, etc.).'
        ''',
        representor=dtype_repr,
        )
    )
