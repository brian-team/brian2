'''
Module implementing the "example" `CodeObject`
'''
from brian2.codegen.codeobject import CodeObject
from brian2.codegen.generators.base import CodeGenerator

__all__ = ['ExamppleCodeObject']

# dummy do-nothing classes


class DummyTemplate():

    def __init__(self, name):
        self.name = name
        self.iterate_all = False
        self.allows_scalar_write = True

    def __call__(self, d):
        return d


class MyClass(object):
    def __init__(self):
        pass

    def __getattr__(self, item):
        return DummyTemplate(item)


class ExampleCodeObject(CodeObject):
    '''
    C++ standalone code object
    
    The ``code`` should be a `~brian2.codegen.templates.MultiTemplate`
    object with two macros defined, ``main`` (for the main loop code) and
    ``support_code`` for any support code (e.g. function definitions).
    '''
    templater = MyClass()
    generator_class = CodeGenerator  # not used
