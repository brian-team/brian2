import os

from brian2.core.variables import Variable, Subexpression
from brian2.codegen.codeobject import CodeObject
from brian2.codegen.templates import Templater
from brian2.codegen.languages.cpp_lang import CPPLanguage

__all__ = ['CPPStandaloneCodeObject']


class CPPStandaloneCodeObject(CodeObject):
    '''
    C++ standalone code object
    
    The ``code`` should be a `~brian2.codegen.languages.templates.MultiTemplate`
    object with two macros defined, ``main`` (for the main loop code) and
    ``support_code`` for any support code (e.g. function definitions).
    '''
    templater = Templater('brian2.devices.cpp_standalone')
    language = CPPLanguage()

    def variables_to_namespace(self):
        # We only copy constant scalar values to the namespace here
        for varname, var in self.variables.iteritems():
            if var.constant and var.scalar:
                self.namespace[varname] = var.get_value()

    def run(self):
        raise RuntimeError("Cannot run in C++ standalone mode")
