'''
Module implementing the C++ "standalone" `CodeObject`
'''
from brian2.codegen.codeobject import CodeObject
from brian2.codegen.templates import Templater
from brian2.codegen.languages.cpp_lang import CPPLanguage
from brian2.devices.device import get_device
from brian2.codegen.languages.cpp_lang import c_data_type

__all__ = ['CPPStandaloneCodeObject']


class CPPStandaloneCodeObject(CodeObject):
    '''
    C++ standalone code object
    
    The ``code`` should be a `~brian2.codegen.languages.templates.MultiTemplate`
    object with two macros defined, ``main`` (for the main loop code) and
    ``support_code`` for any support code (e.g. function definitions).
    '''
    templater = Templater('brian2.devices.cpp_standalone',
                          env_globals={'c_data_type': c_data_type})
    language = CPPLanguage()

    def __call__(self, **kwds):
        return self.run()

    def run(self):
        get_device().main_queue.append(('run_code_object', (self,)))
