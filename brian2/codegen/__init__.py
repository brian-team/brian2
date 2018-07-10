'''
Package providing the code generation framework.
'''
# Import the runtime packages so that they can register their preferences
from .runtime import *
import _prefs
import cpp_prefs as _cpp_prefs

__all__ = ['NumpyCodeObject', 'WeaveCodeObject', 'CythonCodeObject']
