import numpy

from ...codeobject import constant_or_scalar
from ...templates import Templater
from ..cython_rt import CythonCodeObject
from ...generators.GSLcython_generator import GSLCythonCodeGenerator
from ...generators.cython_generator import get_cpp_dtype, get_numpy_dtype

__all__ = ['GSLCythonCodeObject']

class GSLCythonCodeObject(CythonCodeObject):

    templater = Templater('brian2.codegen.runtime.GSLcython_rt', '.pyx',
                          env_globals={'cpp_dtype': get_cpp_dtype,
                                       'numpy_dtype': get_numpy_dtype,
                                       'dtype': numpy.dtype,
                                       'constant_or_scalar': constant_or_scalar})

    generator_class = GSLCythonCodeGenerator
