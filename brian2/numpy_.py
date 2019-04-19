'''
A dummy package to allow importing numpy and the unit-aware replacements of
numpy functions without having to know which functions are overwritten.

This can be used for example as ``import brian2.numpy_ as np``
'''
from __future__ import absolute_import
from numpy import *
from brian2.units.unitsafefunctions import *

# These will not be imported with a wildcard import to not overwrite the
# builtin names (mimicking the numpy behaviour)
try:
    # Python 3
    from builtins import bool, int, float, complex, object, bytes, str
except ImportError:
    # Python 2
    from __builtin__ import bool, int, long, float, complex, object, unicode, str
from numpy.core import round, abs, max, min

import numpy
import brian2.units.unitsafefunctions as brian2_functions
__all__ = []
__all__.extend(numpy.__all__)
__all__.extend(brian2_functions.__all__)
