import os
import sys

import numpy

from brian2.core.variables import (DynamicArrayVariable, ArrayVariable,
                                   AuxiliaryVariable, Subexpression)
from brian2.core.preferences import prefs, BrianPreference
from brian2.utils.logger import get_logger
from brian2.utils.stringtools import get_identifiers

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
        ),
    cache_dir = BrianPreference(
        default=None,
        validator=lambda x: x is None or isinstance(x, basestring),
        docs='''
        Location of the cache directory for Cython files. By default,
        will be stored in a ``brian_extensions`` subdirectory
        where Cython inline stores its temporary files
        (the result of ``get_cython_cache_dir()``).
        '''
        ),
    )


class CythonCodeObject(NumpyCodeObject):
    '''
    Execute code using Cython.
    '''
    templater = Templater('brian2.codegen.runtime.cython_rt', '.pyx',
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
        self.compiled_code = cython_extension_manager.create_extension(
            self.code,
            libraries=self.libraries,
            extra_compile_args=self.extra_compile_args,
            extra_link_args=self.extra_link_args,
            include_dirs=self.include_dirs,
            library_dirs=self.library_dirs,
            compiler=self.compiler,
            owner_name=self.owner.name+'_'+self.template_name,
            )
        
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

        # Get all identifiers in the code -- note that this is not a smart
        # function, it will get identifiers from strings, comments, etc. This
        # is not a problem here, since we only use this list to filter out
        # things. If we include something incorrectly, this only means that we
        # will pass something into the namespace unnecessarily.
        all_identifiers = get_identifiers(self.code)
        # Filter out all unneeded objects
        self.namespace = {k: v for k, v in self.namespace.iteritems()
                          if k in all_identifiers}

        # There is one type of objects that we have to inject into the
        # namespace with their current value at each time step: dynamic
        # arrays that change in size during runs, where the size change is not
        # initiated by the template itself
        for name, var in self.variables.iteritems():
            if (isinstance(var, DynamicArrayVariable) and
                    var.needs_reference_update):
                array_name = self.device.get_array_name(var, self.variables)
                if array_name in self.namespace:
                    self.nonconstant_values.append((array_name, var.get_value))
                if '_num'+name in self.namespace:
                    self.nonconstant_values.append(('_num'+name, var.get_len))

    def update_namespace(self):
        # update the values of the non-constant values in the namespace
        for name, func in self.nonconstant_values:
            self.namespace[name] = func()


codegen_targets.add(CythonCodeObject)
