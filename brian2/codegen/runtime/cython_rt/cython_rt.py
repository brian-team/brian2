import cython
import numpy

from brian2.core.variables import (DynamicArrayVariable, ArrayVariable,
                                   AttributeVariable)
from brian2.core.preferences import brian_prefs, BrianPreference
from brian2.core.functions import DEFAULT_FUNCTIONS, FunctionImplementation, Function

from ...codeobject import CodeObject
from ..numpy_rt import NumpyCodeObject
from ...templates import Templater
from ...generators.cython_generator import CythonCodeGenerator
from ...targets import codegen_targets
from .extension_manager import cython_extension_manager

__all__ = ['CythonCodeObject']


class CythonCodeObject(NumpyCodeObject):
    '''
    '''
    templater = Templater('brian2.codegen.runtime.cython_rt',
                          env_globals={})
    generator_class = CythonCodeGenerator
    class_name = 'cython'

    def __init__(self, owner, code, variables, name='cython_code_object*'):
        super(CythonCodeObject, self).__init__(owner, code, variables, name=name)

    def compile(self):
        self.compiled_code = cython_extension_manager.create_extension(self.code)
        
    def run(self):
        self.compiled_code.main(self.namespace)


codegen_targets.add(CythonCodeObject)
