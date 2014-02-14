'''
Module implementing the C++ "standalone" `CodeObject`
'''
from brian2.codegen.codeobject import CodeObject
from brian2.codegen.targets import codegen_targets
from brian2.codegen.templates import Templater
from brian2.codegen.generators.cpp_generator import (CPPCodeGenerator,
                                                     c_data_type)
from brian2.devices.device import get_device

__all__ = ['CPPStandaloneCodeObject']


class CPPStandaloneCodeObject(CodeObject):
    '''
    C++ standalone code object
    
    The ``code`` should be a `~brian2.codegen.templates.MultiTemplate`
    object with two macros defined, ``main`` (for the main loop code) and
    ``support_code`` for any support code (e.g. function definitions).
    '''
    templater = Templater('brian2.devices.cpp_standalone',
                          env_globals={'c_data_type': c_data_type})
    generator_class = CPPCodeGenerator

    def __call__(self, **kwds):
        return self.run()

    def run(self):
        get_device().main_queue.append(('run_code_object', (self,)))

codegen_targets.add(CPPStandaloneCodeObject)
