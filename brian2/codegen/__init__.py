"""
Package providing the code generation framework.
"""
# Import the runtime packages so that they can register their preferences

from .runtime import *
from . import _prefs
from . import cpp_prefs as _cpp_prefs

__all__ = ["NumpyCodeObject", "CythonCodeObject"]
