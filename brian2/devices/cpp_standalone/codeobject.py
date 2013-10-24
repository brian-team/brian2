'''
Module implementing the C++ "standalone" `CodeObject`
'''
from brian2.core.variables import AttributeVariable
from brian2.codegen.codeobject import CodeObject
from brian2.codegen.templates import Templater
from brian2.codegen.languages.cpp_lang import CPPLanguage
from brian2.devices.device import get_device
from brian2.codegen.runtime.weave_rt.weave_rt import weave_data_type

__all__ = ['CPPStandaloneCodeObject']


class CPPStandaloneCodeObject(CodeObject):
    '''
    C++ standalone code object
    
    The ``code`` should be a `~brian2.codegen.languages.templates.MultiTemplate`
    object with two macros defined, ``main`` (for the main loop code) and
    ``support_code`` for any support code (e.g. function definitions).
    '''
    templater = Templater('brian2.devices.cpp_standalone',
                          env_globals={'c_data_type': weave_data_type})
    language = CPPLanguage()

    def variables_to_namespace(self):
        # We only copy constant scalar values to the namespace here
        for varname, var in self.variables.iteritems():
            if not isinstance(var, AttributeVariable) and var.constant and var.scalar:
                self.namespace[varname] = var.get_value()

    def run(self):
        get_device().main_queue.append(('run_code_object', (self,)))
