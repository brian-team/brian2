"""
A dummy package to allow importing numpy and the unit-aware replacements of
numpy functions without having to know which functions are overwritten.

This can be used for example as ``import brian2.numpy_ as np``
"""

# isort:skip_file
# flake8: noqa

from numpy import *
from brian2.units.unitsafefunctions import *

# These will not be imported with a wildcard import to not overwrite the
# builtin names (mimicking the numpy behaviour)
from builtins import bool, float, complex

from numpy import round, abs, max, min

import numpy
import brian2.units.unitsafefunctions as brian2_functions

__all__ = []
__all__.extend(numpy.__all__)
__all__.extend(brian2_functions.__all__)
