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
    templater = Templater(os.path.join(os.path.split(__file__)[0],
                                       'templates'))
    language = CPPLanguage()

    def variables_to_namespace(self):

        # Variables can refer to values that are either constant (e.g. dt)
        # or change every timestep (e.g. t). We add the values of the
        # constant variables here and add the names of non-constant variables
        # to a list

        # A list containing tuples of name and a function giving the value
        self.nonconstant_values = []

        for name, var in self.variables.iteritems():
            if isinstance(var, Variable) and not isinstance(var, Subexpression):
                if not var.constant:
                    self.nonconstant_values.append((name, var.get_value))
                    if not var.scalar:
                        self.nonconstant_values.append(('_num' + name,
                                                        var.get_len))
                else:
                    try:
                        value = var.get_value()
                    except TypeError:  # A dummy Variable without value
                        continue
                    self.namespace[name] = value
                    # if it is a type that has a length, add a variable called
                    # '_num'+name with its length
                    if not var.scalar:
                        self.namespace['_num' + name] = var.get_len()

    def run(self):
        raise RuntimeError("Cannot run in C++ standalone mode")
