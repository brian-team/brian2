'''
Module providing `WeaveCodeObject`.
'''

import numpy

try:
    from scipy import weave
    from scipy.weave.c_spec import num_to_c_types
except ImportError:
    # No weave for Python 3
    weave = None

from brian2.core.variables import (DynamicArrayVariable, ArrayVariable,
                                   AttributeVariable, AuxiliaryVariable,
                                   Subexpression)
from brian2.core.preferences import brian_prefs, BrianPreference
from brian2.core.functions import DEFAULT_FUNCTIONS

from ...codeobject import CodeObject
from ...templates import Templater
from ...generators.cpp_generator import CPPCodeGenerator
from ...targets import codegen_targets

__all__ = ['WeaveCodeObject', 'WeaveCodeGenerator']

# Preferences
brian_prefs.register_preferences(
    'codegen.runtime.weave',
    'Weave runtime codegen preferences',
    compiler = BrianPreference(
        default='gcc',
        validator=lambda pref: pref=='gcc',
        docs='''
        Compiler to use for weave.
        '''
        ),
    extra_compile_args = BrianPreference(
        default=['-w', '-O3', '-ffast-math'],
        docs='''
        Extra compile arguments to pass to compiler
        '''
        ),
    include_dirs = BrianPreference(
        default=[],
        docs='''
        Include directories to use.
        '''
        )
    )


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
    templater = Templater('brian2.codegen.runtime.weave_rt',
                          env_globals={'c_data_type': weave_data_type,
                                       'dtype': numpy.dtype})
    generator_class = WeaveCodeGenerator
    class_name = 'weave'

    def __init__(self, owner, code, variables, name='weave_code_object*'):
        from brian2.devices.device import get_device
        self.device = get_device()
        self.namespace = {'_owner': owner}
        super(WeaveCodeObject, self).__init__(owner, code, variables, name=name)
        self.compiler = brian_prefs['codegen.runtime.weave.compiler']
        self.extra_compile_args = brian_prefs['codegen.runtime.weave.extra_compile_args']
        self.include_dirs = brian_prefs['codegen.runtime.weave.include_dirs']
        self.python_code_namespace = {'_owner': owner}
        self.variables_to_namespace()

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
                self.nonconstant_values.append((self.device.get_array_name(var,
                                                                           self.variables),
                                                var.get_value))
                self.nonconstant_values.append(('_num'+name, var.get_len))

    def update_namespace(self):
        # update the values of the non-constant values in the namespace
        for name, func in self.nonconstant_values:
            self.namespace[name] = func()
            
    def compile(self):
        CodeObject.compile(self)
        if hasattr(self.code, 'python_pre'):
            self.compiled_python_pre = compile(self.code.python_pre, '(string)', 'exec')
        if hasattr(self.code, 'python_post'):
            self.compiled_python_post = compile(self.code.python_post, '(string)', 'exec')

    def run(self):
        if hasattr(self, 'compiled_python_pre'):
            exec self.compiled_python_pre in self.python_code_namespace
        ret_val = weave.inline(self.code.main, self.namespace.keys(),
                               local_dict=self.namespace,
                               support_code=self.code.support_code,
                               compiler=self.compiler,
                               extra_compile_args=self.extra_compile_args,
                               include_dirs=self.include_dirs)
        if hasattr(self, 'compiled_python_post'):
            exec self.compiled_python_post in self.python_code_namespace
        return ret_val

codegen_targets.add(WeaveCodeObject)


# Use a special implementation for the randn function that makes use of numpy's
# randn
randn_code = {'support_code': '''
        #define BUFFER_SIZE 1024
        // A randn() function that returns a single random number. Internally
        // it asks numpy's randn function for N (e.g. the number of neurons)
        // random numbers at a time and then returns one number from this
        // buffer.
        // It needs a reference to the numpy_randn object (the original numpy
        // function), because this is otherwise only available in
        // compiled_function (where is is automatically handled by weave).
        //
        double _call_randn(py::object& numpy_randn) {
            static PyArrayObject *randn_buffer = NULL;
            static double *buf_pointer = NULL;
            static npy_int curbuffer = 0;
            if(curbuffer==0)
            {
                if(randn_buffer) Py_DECREF(randn_buffer);
                py::tuple args(1);
                args[0] = BUFFER_SIZE;
                randn_buffer = (PyArrayObject *)PyArray_FromAny(numpy_randn.call(args), NULL, 1, 1, 0, NULL);
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
        ''', 'hashdefine_code': '#define _randn(_vectorisation_idx) _call_randn(_python_randn)'}
DEFAULT_FUNCTIONS['randn'].implementations.add_implementation(WeaveCodeObject,
                                                              code=randn_code,
                                                              name='_randn')
