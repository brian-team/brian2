'''
A dummy package to allow importing numpy and the unit-aware replacements of
numpy functions without having to know which functions are overwritten.

This can be used for examples as ``import brian2.numpy_ as np``
'''
from numpy import *
from brian2.units.unitsafefunctions import *
