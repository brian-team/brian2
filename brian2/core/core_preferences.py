'''
Definitions, documentation, default values and validation functions for core
Brian preferences.
'''
from numpy import float64, int32

from brian2.core.preferences import BrianPreference, brian_prefs
from brian2.units.stdunits import ms


def dtype_repr(dtype):
    return dtype.__name__

brian_prefs.register_preferences('core', 'Core Brian preferences',
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
        Default dtype for all arrays of integer scalars'
        ''',
        representor=dtype_repr,
        ),
    default_dt=BrianPreference(
        default=0.1*ms,
        docs='''
        Default time step'
        '''
        )
    )
