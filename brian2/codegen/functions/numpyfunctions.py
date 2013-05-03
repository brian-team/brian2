import sympy
from sympy import Function as sympy_Function
from sympy.core import power as sympy_power
from sympy.core import mod as sympy_mod
import numpy as np
from numpy.random import randn

import brian2.units.unitsafefunctions as unitsafe

from .base import Function

__all__ = ['RandnFunction', 'FunctionWrapper', 'DEFAULT_FUNCTIONS']

class RandnFunction(Function):
    '''
    A specifier for the randn function, allowing its use both in Python and
    C++ code (e.g. for stochastic variables). In Python, a `randn()` call will
    return `N` random numbers (e.g. the size of the `NeuronGroup`), in C++ it
    will return a single number.
    
    Parameters
    ----------
    N : int
        The number of random numbers generated at a time.
    '''
    def __init__(self, N):
        Function.__init__(self, pyfunc=randn)
        self.N = int(N)
    
    def __call__(self):
        return randn(self.N)    
    
    def code_cpp(self, language, var):
        
        support_code = '''
        #define BUFFER_SIZE %N%
        // A randn() function that returns a single random number. Internally
        // it asks numpy's randn function for N (e.g. the number of neurons)
        // random numbers at a time and then returns one number from this
        // buffer.
        // It needs a reference ot the _randn object (the original numpy
        // function), because this is otherwise only available in
        // compiled_function (where is is automatically handled by weave).
        // 
        double randn(py::object& _randn) {        
            static PyArrayObject *randn_buffer = NULL;
            static double *buf_pointer = NULL;
            static npy_int curbuffer = 0;
            if(curbuffer==0)
            {
                if(randn_buffer) Py_DECREF(randn_buffer);
                py::tuple args(1);
                args[0] = BUFFER_SIZE;
                randn_buffer = (PyArrayObject *)PyArray_FromAny(_randn.call(args), NULL, 1, 1, 0, NULL);
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
        '''.replace('%VAR%', var).replace('%N%', str(self.N))

        hashdefine_code = '''
        #define randn() randn(_randn)
        '''.replace('%VAR%', var)
        
        return {'support_code': support_code,
                'hashdefine_code': hashdefine_code}
    
    def on_compile_cpp(self, namespace, language, var):
        pass

class FunctionWrapper(Function):
    '''
    Simple wrapper for functions that have exist both in numpy and C++
    (possibly with a different name, for example ``acos`` vs. ``arccos``).
    
    Parameters
    ----------
    pyfunc : function
        The numpy function (or its unitsafe wrapper)
    py_name : str, optional
        The name of the python function, in case it is not unambiguously
        defined by `pyfunc`. For example, the ``abs`` function in numpy is
        actually named ``absolute`` but we want to use the name ``abs``.
    cpp_name : str, optional
        The name of the corresponding function in C++, in case it is different.        
    sympy_func : sympy function, optional
        The corresponding sympy function, if it exists. 
    '''
    # TODO: How to make this easily extendable for other languages?
    def __init__(self, pyfunc, py_name=None, cpp_name=None, sympy_func=None):
        Function.__init__(self, pyfunc, sympy_func)
        if py_name is None:
            py_name = pyfunc.__name__
        self.py_name = py_name
        self.cpp_name = cpp_name
        
    def code_cpp(self, language, var):
        if self.cpp_name is None:
            hashdefine_code = ''
        else:
            hashdefine_code = '#define {python_name} {cpp_name}'.format(python_name=self.py_name,
                                                                        cpp_name=self.cpp_name)
        return {'support_code': '',
                'hashdefine_code': hashdefine_code}

def _get_default_functions():

    # sympy does not have a log10 function, so let's define one
    class log10(sympy_Function):        
        nargs = 1
        
        @classmethod
        def eval(cls, args):
            return sympy.functions.elementary.exponential.log(args, 10)
            
    functions = { 
                # numpy functions that have the same name in numpy and math.h
                'cos': FunctionWrapper(unitsafe.cos,
                                       sympy_func=sympy.functions.elementary.trigonometric.cos),
                'sin': FunctionWrapper(unitsafe.sin,
                                       sympy_func=sympy.functions.elementary.trigonometric.sin),
                'tan': FunctionWrapper(unitsafe.tan,
                                       sympy_func=sympy.functions.elementary.trigonometric.tan),
                'cosh': FunctionWrapper(unitsafe.cosh,
                                        sympy_func=sympy.functions.elementary.hyperbolic.cosh),
                'sinh': FunctionWrapper(unitsafe.sinh,
                                        sympy_func=sympy.functions.elementary.hyperbolic.sinh),
                'tanh': FunctionWrapper(unitsafe.tanh,
                                        sympy_func=sympy.functions.elementary.hyperbolic.tanh),
                'exp': FunctionWrapper(unitsafe.exp,
                                       sympy_func=sympy.functions.elementary.exponential.exp),
                'log': FunctionWrapper(unitsafe.log,
                                       sympy_func=sympy.functions.elementary.exponential.log),
                'log10': FunctionWrapper(unitsafe.log10,
                                         sympy_func=log10),
                'sqrt': FunctionWrapper(np.sqrt,
                                        sympy_func=sympy.functions.elementary.miscellaneous.sqrt),
                'ceil': FunctionWrapper(np.ceil,
                                        sympy_func=sympy.functions.elementary.integers.ceiling),
                'floor': FunctionWrapper(np.floor,
                                         sympy_func=sympy.functions.elementary.integers.floor),     
                # numpy functions that have a different name in numpy and math.h
                'arccos': FunctionWrapper(unitsafe.arccos,
                                          cpp_name='acos',
                                          sympy_func=sympy.functions.elementary.trigonometric.acos),
                'arcsin': FunctionWrapper(unitsafe.arcsin,
                                          cpp_name='asin',
                                          sympy_func=sympy.functions.elementary.trigonometric.asin),
                'arctan': FunctionWrapper(unitsafe.arctan,
                                          cpp_name='atan',
                                          sympy_func=sympy.functions.elementary.trigonometric.atan),                      
                'power': FunctionWrapper(np.power,
                                         cpp_name='pow',
                                         sympy_func=sympy_power.Pow),
                'abs': FunctionWrapper(np.abs, py_name='abs',
                                       cpp_name='fabs',
                                       sympy_func=sympy.functions.elementary.complexes.Abs),
                'mod': FunctionWrapper(np.mod, py_name='mod',
                                       cpp_name='fmod',
                                       sympy_func=sympy_mod.Mod)
                 }
    
    return functions

DEFAULT_FUNCTIONS = _get_default_functions()