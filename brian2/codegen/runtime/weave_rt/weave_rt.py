'''
Module providing `WeaveCodeObject`.
'''
import os
import sys
import numpy

try:
    from scipy import weave
    from scipy.weave.c_spec import num_to_c_types
    from scipy.weave.inline_tools import function_cache
except ImportError:
    try:  # weave as an independent package
        import weave
        from weave.c_spec import num_to_c_types
        from weave.inline_tools import function_cache
    except ImportError:
        # No weave for Python 3
        weave = None

import brian2
from brian2.core.variables import (DynamicArrayVariable, ArrayVariable,
                                   AuxiliaryVariable, Subexpression)
from brian2.core.preferences import prefs
from brian2.core.functions import DEFAULT_FUNCTIONS, Function
from brian2.utils.logger import std_silent, get_logger
from brian2.utils.stringtools import get_identifiers

from ...codeobject import CodeObject, constant_or_scalar
from ...templates import Templater
from ...generators.cpp_generator import CPPCodeGenerator
from ...targets import codegen_targets
from ...cpp_prefs import get_compiler_and_args

__all__ = ['WeaveCodeObject', 'WeaveCodeGenerator']


logger = get_logger(__name__)


def weave_data_type(dtype):
    '''
    Gives the C language specifier for numpy data types using weave. For example,
    ``numpy.int32`` maps to ``long`` in C.
    '''
    # this handles the case where int is specified, it will be int32 or int64
    # depending on platform
    if dtype is int:
        dtype = numpy.array([1]).dtype.type
    if dtype is float:
        dtype = numpy.array([1.]).dtype.type
    try:
        dtype = numpy.empty(0, dtype=dtype).dtype.char
    except TypeError:
        raise TypeError('Illegal dtype %r' % dtype)
        
    return num_to_c_types[dtype]


class WeaveCodeGenerator(CPPCodeGenerator):
    def __init__(self, *args, **kwds):
        super(WeaveCodeGenerator, self).__init__(*args, **kwds)
        self.c_data_type = weave_data_type


class WeaveCodeObject(CodeObject):
    '''
    Weave code object
    
    The ``code`` should be a `~brian2.codegen.templates.MultiTemplate`
    object with two macros defined, ``main`` (for the main loop code) and
    ``support_code`` for any support code (e.g. function definitions).
    '''
    templater = Templater('brian2.codegen.runtime.weave_rt', '.cpp',
                          env_globals={'c_data_type': weave_data_type,
                                       'dtype': numpy.dtype,
                                       'constant_or_scalar': constant_or_scalar})
    generator_class = WeaveCodeGenerator
    class_name = 'weave'

    def __init__(self, owner, code, variables, variable_indices,
                 template_name, template_source, name='weave_code_object*'):
        from brian2.devices.device import get_device
        self.device = get_device()
        self._done_first_run = False
        self.namespace = {'_owner': owner}
        super(WeaveCodeObject, self).__init__(owner, code, variables,
                                              variable_indices,
                                              template_name, template_source,
                                              name=name)
        self.compiler, self.extra_compile_args = get_compiler_and_args()
        self.define_macros = list(prefs['codegen.cpp.define_macros'])
        if self.compiler == 'msvc':
            self.define_macros.extend([
                ('INFINITY', '(std::numeric_limits<double>::infinity())'),
                ('NAN', '(std::numeric_limits<double>::quiet_NaN())'),
                ('M_PI', '3.14159265358979323846')
            ])
        self.extra_link_args = list(prefs['codegen.cpp.extra_link_args'])
        self.include_dirs = list(prefs['codegen.cpp.include_dirs'])
        self.include_dirs += [os.path.join(sys.prefix, 'include')]
        # TODO: We should probably have a special folder just for header
        # files that are shared between different codegen targets
        import brian2.synapses as synapses
        synapses_dir = os.path.abspath(os.path.dirname(synapses.__file__))
        brian2dir, _ = os.path.split(brian2.__file__)
        rkdir = os.path.abspath(os.path.join(brian2dir, 'utils',
                                             'random', 'randomkit'))

        self.include_dirs.append(synapses_dir)
        self.include_dirs.append(rkdir)
        self.library_dirs = list(prefs['codegen.cpp.library_dirs']) + [rkdir]
        self.runtime_library_dirs = list(prefs['codegen.cpp.runtime_library_dirs'])
        self.extra_objects = []
        if sys.platform == 'linux2':
            self.runtime_library_dirs += [rkdir]
        elif sys.platform == 'darwin':
            self.extra_objects += [os.path.join(rkdir, 'librandomkit.so')]
        self.libraries = list(prefs['codegen.cpp.libraries'])
        if sys.platform == 'win32':
            self.libraries.append('advapi32')  # needed for randomkit
            self.libraries.append('librandomkit.pyd')
        else:
            self.libraries.append('randomkit')
        self.headers = (['<algorithm>', '<limits>',
                         '"stdint_compat.h"', '"randomkit.h"'] +
                        prefs['codegen.cpp.headers'])
        self.annotated_code = self.code.main+'''
/*
The following code is just compiler options for the call to weave.inline.
By including them here, we force a recompile if the compiler options change,
which is a good thing (e.g. switching -ffast-math on and off).

support_code:
{self.code.support_code}

compiler: {self.compiler}
define_macros: {self.define_macros}
extra_compile_args: {self.extra_compile_args}
extra_link_args: {self.extra_link_args}
include_dirs: {self.include_dirs}
library_dirs: {self.library_dirs}
runtime_library_dirs: {self.runtime_library_dirs}
libraries: {self.libraries}
*/
        '''.format(self=self)

        self.python_code_namespace = {'_owner': owner}
        self.variables_to_namespace()

    @staticmethod
    def is_available():
        try:
            with std_silent(False):
                compiler, extra_compile_args = get_compiler_and_args()
                weave.inline('int x=0;', [],
                             compiler=compiler,
                             headers=['<algorithm>', '<limits>'],
                             extra_compile_args=extra_compile_args,
                             extra_link_args=prefs['codegen.cpp.extra_link_args'],
                             library_dirs=prefs['codegen.cpp.library_dirs'],
                             include_dirs=prefs['codegen.cpp.include_dirs'],
                             verbose=0)
                return True
        except Exception as ex:
            logger.warn(('Cannot use weave, a test compilation '
                         'failed: %s (%s)' % (str(ex),
                                              ex.__class__.__name__)) ,
                        'failed_compile_test')
            return False

    def variables_to_namespace(self):

        # Variables can refer to values that are either constant (e.g. dt)
        # or change every timestep (e.g. t). We add the values of the
        # constant variables here and add the names of non-constant variables
        # to a list

        # A list containing tuples of name and a function giving the value
        self.nonconstant_values = []

        for name, var in self.variables.iteritems():
            if isinstance(var, (AuxiliaryVariable, Subexpression, Function)):
                continue
            try:
                value = var.get_value()
            except (TypeError, AttributeError):
                # A dummy Variable without value or a an object that is accessed
                # with Python's C API directly
                self.namespace[name] = var
                continue

            if isinstance(var, ArrayVariable):
                self.namespace[self.device.get_array_name(var,
                                                            self.variables)] = value
                self.namespace['_num'+name] = var.get_len()
                # if var.scalar and var.constant:
                #     self.namespace[name] = value.item()
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
        all_identifiers = reduce(lambda s, c: s | get_identifiers(c),
                                 self.code.values(), set())
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
            
    def compile(self):
        CodeObject.compile(self)
        if hasattr(self.code, 'python_pre'):
            self.compiled_python_pre = compile(self.code.python_pre, '(string)', 'exec')
        else:
            self.compiled_python_pre = None
        if hasattr(self.code, 'python_post'):
            self.compiled_python_post = compile(self.code.python_post, '(string)', 'exec')
        else:
            self.compiled_python_post = None

    def run(self):
        if self.compiled_python_pre is not None:
            exec self.compiled_python_pre in self.python_code_namespace
        if self._done_first_run:
            ret_val = self._compiled_func(self.namespace, {})
        else:
            self._inline_args = (self.annotated_code, self.namespace.keys())
            self._inline_kwds = dict(
                local_dict=self.namespace,
                support_code=self.code.support_code,
                compiler=self.compiler,
                headers=self.headers,
                define_macros=self.define_macros,
                libraries=self.libraries,
                extra_compile_args=self.extra_compile_args,
                extra_link_args=self.extra_link_args,
                extra_objects=self.extra_objects,
                include_dirs=self.include_dirs,
                library_dirs=self.library_dirs,
                runtime_library_dirs=self.runtime_library_dirs,
                verbose=2)
            with std_silent():
                ret_val = weave.inline(*self._inline_args, **self._inline_kwds)
            self._compiled_func = function_cache[self.annotated_code]
            self._done_first_run = True
        if self.compiled_python_post is not None:
            exec self.compiled_python_post in self.python_code_namespace
        return ret_val

if weave is not None:
    codegen_targets.add(WeaveCodeObject)


# Use RandomKit for random number generation (same algorithm as numpy)
# Note that we create a new random state for each codeobject, but this
# is seeded with a few hundred bytes from OS-level entropy generators
# and so the random numbers produced will be independent. If these
# RNGs aren't able to produce enough data to seed the RNG it will raise
# an error.
randn_code = {'support_code': '''
        rk_state *_mtrandstate_randn = NULL;
        inline double _randn(const int _vectorisation_idx) {
            if(_mtrandstate_randn==NULL) {
                _mtrandstate_randn = new rk_state;
                rk_error errcode = rk_randomseed(_mtrandstate_randn);
                if(errcode)
                {
                    PyErr_SetString(PyExc_RuntimeError, "Cannot initialise random state");
                    throw 1;
                }
            }
            return rk_gauss(_mtrandstate_randn);
        }
        '''}
DEFAULT_FUNCTIONS['randn'].implementations.add_implementation(WeaveCodeObject,
                                                              code=randn_code,
                                                              name='_randn')

rand_code = {'support_code': '''
        rk_state *_mtrandstate_rand = NULL;
        inline double _rand(const int _vectorisation_idx) {
            if(_mtrandstate_rand==NULL) {
                _mtrandstate_rand = new rk_state;
                rk_error errcode = rk_randomseed(_mtrandstate_rand);
                if(errcode)
                {
                    PyErr_SetString(PyExc_RuntimeError, "Cannot initialise random state");
                    throw 1;
                }
            }
            return rk_double(_mtrandstate_rand);
        }
        '''}
DEFAULT_FUNCTIONS['rand'].implementations.add_implementation(WeaveCodeObject,
                                                             code=rand_code,
                                                             name='_rand')
