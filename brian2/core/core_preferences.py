'''
Definitions, documentation, default values and validation functions for core
Brian preferences.
'''
from numpy import dtype

from brian2.core.preferences import BrianPreference, register_preferences

register_preferences('core', 'Core Brian preferences',
    default_scalar_dtype=BrianPreference(
        default='float64',
        docs='''
        Default dtype for all arrays of scalars (state variables, weights, etc.).'
        ''',
        validator=dtype,
        ),
    delete_log_on_exit=BrianPreference(
        default='True',
        docs=    '''
        Whether to delete the log and script file on exit.
        
        If set to ``True`` (the default), log files (and the copy of the main
        script) will be deleted after the brian process has exited, unless an
        uncaught exception occured. If set to ``False``, all log files will be kept.
        ''',
        validator=bool,
        ),
    weave_compiler=BrianPreference(
        default='gcc',
        docs='''
        The compiler name to use for ``scipy.weave``.
        
        On Linux platforms, gcc is usually available. On Windows, it can be made
        available as part of the mingw and cygwin packages, and also included in
        some Python distributions such as EPD and Python(x,y).
        ''',
        validator=str,
        ),
    )
