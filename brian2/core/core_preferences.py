"""
Definitions, documentation, default values and validation functions for core
Brian preferences.
"""

from numpy import float32, float64, int32

from brian2.core.preferences import BrianPreference, prefs

__all__ = []


def dtype_repr(dtype):
    return dtype.__name__


def default_float_dtype_validator(dtype):
    return dtype in [float32, float64]


def outdated_dependency_error_validator(value):
    if value is not None:
        import warnings

        warnings.warn(
            "The 'core.outdated_dependency_error' preference is no longer "
            "used and will be removed in a future version. Brian2 now "
            "relies on package managers for dependency management.",
            UserWarning,
            stacklevel=2,
        )
    return True  # We always accept the value user provides, but warn them


prefs.register_preferences(
    "core",
    "Core Brian preferences",
    default_float_dtype=BrianPreference(
        default=float64,
        docs="""
        Default dtype for all arrays of scalars (state variables, weights, etc.).
        """,
        representor=dtype_repr,
        validator=default_float_dtype_validator,
    ),
    default_integer_dtype=BrianPreference(
        default=int32,
        docs="""
        Default dtype for all arrays of integer scalars.
        """,
        representor=dtype_repr,
    ),
    outdated_dependency_error=BrianPreference(
        default=None,
        docs="""
        **DEPRECATED**: This preference is no longer used. Brian2 now relies on
        package managers (pip, conda) for dependency management instead of runtime
        version checking. Setting this preference has no effect and it will be
        removed in a future version.

        Previously controlled whether to raise an error for outdated dependencies
        (``True``) or just a warning (``False``).
        """,
        validator=outdated_dependency_error_validator,
    ),
    stop_on_keyboard_interrupt=BrianPreference(
        default=True,
        docs="""
        Whether to "gracefully" stop a simulation after pressing Ctrl+C (defaults to
        ``True``). Note that pressing Ctrl+C a second time will force the usual
        interruption mechanism.
        """,
    ),
)

prefs.register_preferences(
    "legacy",
    "Preferences to enable legacy behaviour",
    refractory_timing=BrianPreference(
        default=False,
        docs="""
        Whether to use the semantics for checking the refractoriness condition
        that were in place up until (including) version 2.1.2. In that
        implementation, refractory periods that were multiples of dt could lead
        to a varying number of refractory timesteps due to the nature of
        floating point comparisons). This preference is only provided for exact
        reproducibility of previously obtained results, new simulations should
        use the improved mechanism which uses a more robust mechanism to
        convert refractoriness into timesteps. Defaults to ``False``.
        """,
    ),
)
