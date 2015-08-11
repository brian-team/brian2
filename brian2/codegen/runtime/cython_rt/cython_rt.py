import os
import sys

import numpy

from brian2.core.variables import (DynamicArrayVariable, ArrayVariable,
                                   AttributeVariable, AuxiliaryVariable,
                                   Subexpression)
from brian2.core.preferences import prefs, BrianPreference
from brian2.utils.logger import get_logger

from ..numpy_rt import NumpyCodeObject
from ...templates import Templater
from ...generators.cython_generator import (CythonCodeGenerator, get_cpp_dtype,
                                            get_numpy_dtype)
from ...targets import codegen_targets
from ...cpp_prefs import get_compiler_and_args
from .extension_manager import cython_extension_manager

__all__ = ['CythonCodeObject']


logger = get_logger(__name__)

# Preferences
prefs.register_preferences(
    'codegen.runtime.cython',
    'Cython runtime codegen preferences',
    multiprocess_safe = BrianPreference(
        default=True,
        docs='''
        Whether to use a lock file to prevent simultaneous write access
        to cython .pyx and .so files.
        '''
        )
    )


class CythonCodeObject(NumpyCodeObject):
    '''
    Execute code using Cython.
    '''
    templater = Templater('brian2.codegen.runtime.cython_rt',
                          env_globals={'cpp_dtype': get_cpp_dtype,
                                       'numpy_dtype': get_numpy_dtype,
                                       'dtype': numpy.dtype})
    generator_class = CythonCodeGenerator
    class_name = 'cython'

    def __init__(self, owner, code, variables, variable_indices,
                 template_name, template_source, name='cython_code_object*'):
        super(CythonCodeObject, self).__init__(owner, code, variables,
                                               variable_indices,
                                               template_name, template_source,
                                               name=name)
        self.compiler, self.extra_compile_args = get_compiler_and_args()
        self.extra_link_args = list(prefs['codegen.cpp.extra_link_args'])
        self.include_dirs = list(prefs['codegen.cpp.include_dirs'])
        self.include_dirs += [os.path.join(sys.prefix, 'include')]
        self.library_dirs = list(prefs['codegen.cpp.library_dirs'])
        self.runtime_library_dirs = list(prefs['codegen.cpp.runtime_library_dirs'])
        self.libraries = list(prefs['codegen.cpp.libraries'])

    @staticmethod
    def is_available():
        try:
            compiler, extra_compile_args = get_compiler_and_args()
            code = '''
            def main():
                cdef int x
                x = 0'''
            compiled = cython_extension_manager.create_extension(code,
                                                                 compiler=compiler,
                                                                 extra_compile_args=extra_compile_args,
                                                                 extra_link_args=prefs['codegen.cpp.extra_link_args'],
                                                                 include_dirs=prefs['codegen.cpp.include_dirs'],
                                                                 library_dirs=prefs['codegen.cpp.library_dirs'])
            compiled.main()
            return True
        except Exception as ex:
            logger.warn(('Cannot use Cython, a test compilation '
                         'failed: %s (%s)' % (str(ex),
                                              ex.__class__.__name__)) ,
                        'failed_compile_test')
            return False


    def compile(self):
        self.compiled_code = cython_extension_manager.create_extension(self.code,
                                                                       libraries=self.libraries,
                                                                       extra_compile_args=self.extra_compile_args,
                                                                       extra_link_args=self.extra_link_args,
                                                                       include_dirs=self.include_dirs,
                                                                       library_dirs=self.library_dirs,
                                                                       compiler=self.compiler)
        
    def run(self):
        return self.compiled_code.main(self.namespace)

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
                if var.scalar and var.constant:
                    self.namespace[name] = value.item()
            else:
                self.namespace[name] = value

            if isinstance(var, DynamicArrayVariable):
                dyn_array_name = self.generator_class.get_array_name(var,
                                                                    access_data=False)
                self.namespace[dyn_array_name] = self.device.get_value(var,
                                                                       access_data=False)

            # Also provide the Variable object itself in the namespace (can be
            # necessary for resize operations, for example)
            self.namespace['_var_'+name] = var

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
