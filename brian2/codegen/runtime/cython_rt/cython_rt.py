import cython
import numpy

from brian2.core.variables import (DynamicArrayVariable, ArrayVariable,
                                   AttributeVariable, AuxiliaryVariable,
                                   Subexpression)
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
#        print '**** namespace.keys() at runtime for object', self.name+'\n    ' +str(self.namespace.keys())
#        for k, v in self.namespace.items():
#            print '  ', k, v.__class__
        self.compiled_code.main(self.namespace)

    # the following are copied from WeaveCodeObject

    def variables_to_namespace(self):

        # Variables can refer to values that are either constant (e.g. dt)
        # or change every timestep (e.g. t). We add the values of the
        # constant variables here and add the names of non-constant variables
        # to a list

        # A list containing tuples of name and a function giving the value
        self.nonconstant_values = []

        for name, var in self.variables.iteritems():
            if isinstance(var, (AuxiliaryVariable, Subexpression)):
                continue
            try:
                value = var.get_value()
            except (TypeError, AttributeError):
                # A dummy Variable without value or a function
                self.namespace[name] = var
                continue

            if isinstance(var, ArrayVariable):
                self.namespace[self.device.get_array_name(var,
                                                            self.variables)] = value
                self.namespace['_num'+name] = var.get_len()
            else:
                self.namespace[name] = value

            if isinstance(var, DynamicArrayVariable):
                dyn_array_name = self.generator_class.get_array_name(var,
                                                                    access_data=False)
                self.namespace[dyn_array_name] = self.device.get_value(var,
                                                                       access_data=False)

            # There are two kinds of objects that we have to inject into the
            # namespace with their current value at each time step:
            # * non-constant AttributeValue (this might be removed since it only
            #   applies to "t" currently)
            # * Dynamic arrays that change in size during runs (i.e. not
            #   synapses but e.g. the structures used in monitors)
            if isinstance(var, AttributeVariable) and not var.constant:
                self.nonconstant_values.append((name, var.get_value))
                if not var.scalar:
                    self.nonconstant_values.append(('_num'+name, var.get_len))
            elif (isinstance(var, DynamicArrayVariable) and
                  not var.constant_size):
                #print name, self.device.get_array_name(var, self.variable), self.generator_class.get_array_name(var, self.variables)
                self.nonconstant_values.append((self.device.get_array_name(var, True),
                                                var.get_value))
                self.nonconstant_values.append(('_num'+name, var.get_len))

    def update_namespace(self):
        # update the values of the non-constant values in the namespace
        for name, func in self.nonconstant_values:
            self.namespace[name] = func()


codegen_targets.add(CythonCodeObject)
