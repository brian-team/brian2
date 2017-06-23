from ..cython_rt import CythonCodeObject
from ...generators.GSL_generator import GSLCodeGenerator
from ...generators.cython_generator import CythonCodeGenerator

__all__ = ['GSLCythonCodeObject']

class GSLCythonCodeObject(CythonCodeObject):

    templater = CythonCodeObject.templater.derive('brian2.codegen.runtime.GSLcython_rt')

    original_generator_class = CythonCodeGenerator
    generator_class = GSLCodeGenerator
