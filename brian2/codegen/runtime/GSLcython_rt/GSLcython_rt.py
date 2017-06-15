from ..cython_rt import *
from ...generators.GSLcython_generator import GSLCythonCodeGenerator

__all__ = ['GSLCythonCodeObject']

class GSLCythonCodeObject(CythonCodeObject):

    generator_class = GSLCythonCodeGenerator
