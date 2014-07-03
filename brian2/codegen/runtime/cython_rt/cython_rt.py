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
from .modified_inline import modified_cython_inline

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
        CodeObject.compile(self)

    def run(self):
#        cyfunc, args = modified_cython_inline(self.code, locals=self.namespace, globals={})
#        cyfunc.__invoke(*args)
        return modified_cython_inline(self.code, locals=self.namespace, globals={})
        #return cython.inline(self.code, locals=self.namespace, globals={})
        # output variables should land in the variable name _return_values
        if '_return_values' in self.namespace:
            return self.namespace['_return_values']

codegen_targets.add(CythonCodeObject)
