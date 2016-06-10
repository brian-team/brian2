'''
Module implementing the "example" `CodeObject`
'''
from brian2.codegen.codeobject import CodeObject, constant_or_scalar
from brian2.codegen.targets import codegen_targets
from brian2.codegen.templates import Templater
from brian2.codegen.generators.cpp_generator import (CPPCodeGenerator,
                                                     c_data_type)
from brian2.devices.device import get_device
from brian2.core.preferences import prefs
from brian2.core.functions import DEFAULT_FUNCTIONS
from brian2.utils.stringtools import replace

__all__ = ['ExamppleCodeObject']


class ExampleCodeObject(CodeObject):
    '''
    C++ standalone code object
    
    The ``code`` should be a `~brian2.codegen.templates.MultiTemplate`
    object with two macros defined, ``main`` (for the main loop code) and
    ``support_code`` for any support code (e.g. function definitions).
    '''
    templater = Templater('brian2.devices.example', '.py_')
    generator_class = ExampleCodeGenerator

codegen_targets.add(ExampleCodeObject)
