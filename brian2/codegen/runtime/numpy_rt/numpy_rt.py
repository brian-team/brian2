'''
Module providing `NumpyCodeObject`.
'''
import numpy as np

from brian2.core.preferences import brian_prefs, BrianPreference
from brian2.core.variables import (DynamicArrayVariable, ArrayVariable,
                                   AttributeVariable)
from brian2.synapses.spikequeue import SpikeQueue

from ...codeobject import CodeObject

from ...templates import Templater
from ...languages.numpy_lang import NumpyLanguage
from ...targets import codegen_targets

__all__ = ['NumpyCodeObject']

# Preferences
brian_prefs.register_preferences(
    'codegen.runtime.numpy',
    'Numpy runtime codegen preferences',
    discard_units = BrianPreference(
        default=False,
        docs='''
        Whether to change the namespace of user-specifed functions to remove
        units.
        '''
        )
    )


class NumpyCodeObject(CodeObject):
    '''
    Execute code using Numpy
    
    Default for Brian because it works on all platforms.
    '''
    templater = Templater('brian2.codegen.runtime.numpy_rt')
    language = NumpyLanguage()
    class_name = 'numpy'

    def __init__(self, owner, code, namespace, variables, name='numpy_code_object*'):
        # TODO: This should maybe go somewhere else
        namespace['logical_not'] = np.logical_not
        CodeObject.__init__(self, owner, code, namespace, variables, name=name)
        namespace['_owner'] = self.owner

    def variables_to_namespace(self):
        # Variables can refer to values that are either constant (e.g. dt)
        # or change every timestep (e.g. t). We add the values of the
        # constant variables here and add the names of non-constant variables
        # to a list

        # A list containing tuples of name and a function giving the value
        self.nonconstant_values = []

        for name, var in self.variables.iteritems():
            try:
                value = var.get_value()
            except TypeError:  # A dummy Variable without value or a Subexpression
                continue

            self.namespace[name] = value

            if isinstance(var, ArrayVariable):
                self.namespace[var.arrayname] = value

            if isinstance(var, DynamicArrayVariable):
                self.namespace[var.name+'_object'] = var.get_object()

            # There are two kinds of objects that we have to inject into the
            # namespace with their current value at each time step:
            # * non-constant AttributeValue (this might be removed since it only
            #   applies to "t" currently)
            # * Dynamic arrays that change in size during runs (i.e. not
            #   synapses but e.g. the structures used in monitors)
            if isinstance(var, AttributeVariable) and not var.constant:
                self.nonconstant_values.append((name, var.get_value))
            elif (isinstance(var, DynamicArrayVariable) and
                  not var.constant_size):
                self.nonconstant_values.append((var.arrayname,
                                                var.get_value))

    def update_namespace(self):
        # update the values of the non-constant values in the namespace
        for name, func in self.nonconstant_values:
            self.namespace[name] = func()

    def compile(self):
        super(NumpyCodeObject, self).compile()
        self.compiled_code = compile(self.code, '(string)', 'exec')

    def run(self):
        exec self.compiled_code in self.namespace
        # output variables should land in the variable name _return_values
        if '_return_values' in self.namespace:
            return self.namespace['_return_values']

codegen_targets.add(NumpyCodeObject)
