from ..cython_rt import CythonCodeObject
from ...generators.GSL_generator import GSLCythonCodeGenerator
from ...generators.cython_generator import CythonCodeGenerator

__all__ = ['GSLCythonCodeObject', 'IntegrationError']

class IntegrationError(Exception):
    '''
    Error used to signify that GSL was unable to complete integration (only works for cython)
    '''
    pass

class GSLCythonCodeObject(CythonCodeObject):

    templater = CythonCodeObject.templater.derive('brian2.codegen.runtime.GSLcython_rt')

    # CodeGenerator that is used to do bulk of abstract_code --> language specific translation
    original_generator_class = CythonCodeGenerator
    generator_class = GSLCythonCodeGenerator