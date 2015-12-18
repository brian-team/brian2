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
        synapses_dir = os.path.dirname(synapses.__file__)
        self.include_dirs.append(synapses_dir)
        self.library_dirs = list(prefs['codegen.cpp.library_dirs'])
        self.runtime_library_dirs = list(prefs['codegen.cpp.runtime_library_dirs'])
        self.libraries = list(prefs['codegen.cpp.libraries'])
        self.headers = ['<algorithm>', '<limits>', '"stdint_compat.h"'] + prefs['codegen.cpp.headers']
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
                include_dirs=self.include_dirs,
                library_dirs=self.library_dirs,
                verbose=0)
            with std_silent():
                ret_val = weave.inline(*self._inline_args, **self._inline_kwds)
            self._compiled_func = function_cache[self.annotated_code]
            self._done_first_run = True
        if self.compiled_python_post is not None:
            exec self.compiled_python_post in self.python_code_namespace
        return ret_val

if weave is not None:
    codegen_targets.add(WeaveCodeObject)


# Use a special implementation for the randn function that makes use of numpy's
# randn
randn_code = {'support_code': '''
        #define BUFFER_SIZE 1024
        // A randn() function that returns a single random number. Internally
        // it asks numpy's randn function for BUFFER_SIZE
        // random numbers at a time and then returns one number from this
        // buffer.
        // It needs a reference to the numpy_randn object (the original numpy
        // function), because this is otherwise only available in
        // compiled_function (where is is automatically handled by weave).
        //
        double _randn(const int _vectorisation_idx) {
            // the _vectorisation_idx argument is unused for now, it could in
            // principle be used to get reproducible random numbers when using
            // OpenMP etc.
            static PyArrayObject *randn_buffer = NULL;
            static double *buf_pointer = NULL;
            static npy_int curbuffer = 0;
            if(curbuffer==0)
            {
                if(randn_buffer) Py_DECREF(randn_buffer);
                py::tuple args(1);
                args[0] = BUFFER_SIZE;
                randn_buffer = (PyArrayObject *)PyArray_FromAny(_namespace_numpy_randn.call(args),
                                                                NULL, 1, 1, 0, NULL);
                buf_pointer = (double*)PyArray_GETPTR1(randn_buffer, 0);
            }
            double number = buf_pointer[curbuffer];
            curbuffer = curbuffer+1;
            if (curbuffer == BUFFER_SIZE)
                // This seems to be safer then using (curbuffer + 1) % BUFFER_SIZE, we might run into
                // an integer overflow for big networks, otherwise.
                curbuffer = 0;
            return number;
        }
        '''}
DEFAULT_FUNCTIONS['randn'].implementations.add_implementation(WeaveCodeObject,
                                                              code=randn_code,
                                                              name='_randn',
                                                              namespace={'_numpy_randn': numpy.random.randn})

# Also use numpy for rand
rand_code = {'support_code': '''
        #define BUFFER_SIZE 1024
        // A rand() function that returns a single random number. Internally
        // it asks numpy's rand function for BUFFER_SIZE
        // random numbers at a time and then returns one number from this
        // buffer.
        // It needs a reference to the numpy_rand object (the original numpy
        // function), because this is otherwise only available in
        // compiled_function (where is is automatically handled by weave).
        //
        double _rand(const int _vectorisation_idx) {
            // the _vectorisation_idx argument is unused for now, it could in
            // principle be used to get reproducible random numbers when using
            // OpenMP etc.
            static PyArrayObject *rand_buffer = NULL;
            static double *buf_pointer = NULL;
            static npy_int curbuffer = 0;
            if(curbuffer==0)
            {
                if(rand_buffer) Py_DECREF(rand_buffer);
                py::tuple args(1);
                args[0] = BUFFER_SIZE;
                rand_buffer = (PyArrayObject *)PyArray_FromAny(_namespace_numpy_rand.call(args),
                                                               NULL, 1, 1, 0, NULL);
                buf_pointer = (double*)PyArray_GETPTR1(rand_buffer, 0);
            }
            double number = buf_pointer[curbuffer];
            curbuffer = curbuffer+1;
            if (curbuffer == BUFFER_SIZE)
                // This seems to be safer then using (curbuffer + 1) % BUFFER_SIZE, we might run into
                // an integer overflow for big networks, otherwise.
                curbuffer = 0;
            return number;
        }
        '''}
DEFAULT_FUNCTIONS['rand'].implementations.add_implementation(WeaveCodeObject,
                                                             code=rand_code,
                                                             namespace={'_numpy_rand': numpy.random.rand},
                                                             name='_rand')
