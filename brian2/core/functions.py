import sympy
from sympy import Function as sympy_Function
from sympy.core import mod as sympy_mod
import numpy as np
from numpy.random import randn, rand

import brian2.units.unitsafefunctions as unitsafe

__all__ = ['FunctionWrapper', 'DEFAULT_FUNCTIONS', 'Function',
           'SimpleFunction', 'make_function']


class Function(object):
    def __init__(self, pyfunc, sympy_func=None, arg_units=None,
                 return_unit=None, codes=None):

        self.pyfunc = pyfunc
        self.sympy_func = sympy_func
        if hasattr(pyfunc, '_arg_units'):
            self._arg_units = pyfunc._arg_units
            self._return_unit = pyfunc._return_unit
        else:
            if arg_units is None or return_unit is None:
                raise ValueError(('The given Python function does not specify '
                                  'how it deals with units, need to specify '
                                  '"arg_units" and "return_unit"'))
            self._arg_units = arg_units
            self._return_unit = return_unit
        if codes is None:
            codes = {}
        self.codes = codes

    def code(self, language_id):
        """
        Returns a dict of ``(slot, section)`` values, where ``slot`` is a
        language-specific slot for where to include the string ``section``.

        The list of slot names to use is language-specific.
        """
        if language_id in self.codes:
            return self.codes[language_id]
        elif language_id == 'python' and self.pyfunc is not None:
            return self.pyfunc
        else:
            raise NotImplementedError('Function not implemented for '
                                      'language ' + language_id)

    def __call__(self, *args):
        return self.code('python')(*args)


def make_function(codes, namespace):
    '''
    A simple decorator to extend user-written Python functions to work with code
    generation in other languages.

    You provide a dict ``codes`` of ``(language_id, code)`` pairs and a
    namespace of values to be added to the generated code. The ``code`` should
    be in the format recognised by the language (e.g. dict or string).

    Sample usage::

        @make_function(codes={
            'cpp':{
                'support_code':"""
                    #include<math.h>
                    inline double usersin(double x)
                    {
                        return sin(x);
                    }
                    """,
                'hashdefine_code':'',
                },
            }, namespace={})
        def usersin(x):
            return sin(x)
    '''
    def do_make_user_function(func):
        return Function(func, codes=codes)
    return do_make_user_function


################################################################################
# Standard functions
################################################################################
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
    def __init__(self):
        codes = {'python': lambda vectorisation_idx: randn(len(vectorisation_idx)),
                 'cpp': {'support_code':
        '''
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
        ''',
                         'hashdefine_code': '''
        #define _randn(_vectorisation_idx) _call_randn(_python_randn)
        '''}
        }
        Function.__init__(self, pyfunc=randn, arg_units=[], return_unit=1,
                          codes=codes)


class RandFunction(Function):
    '''
    A specifier for the rand function, allowing its use both in Python and
    C++ code (e.g. for synaptic connectivity). In Python, a
    `rand(vectorisation_idx)` call will return `len(vectorisation_idx)` random
    numbers (e.g. the size of the `NeuronGroup`), in C++ it will return a
    single number.
    '''
    def __init__(self):
        codes = {'python': lambda vectorisation_idx: rand(len(vectorisation_idx)),
                 'cpp': {'support_code':
        '''
        double _rand(int vectorisation_idx)
        {
	        return (double)rand()/RAND_MAX;
        }
        '''
                      }
        }
        Function.__init__(self, pyfunc=rand, arg_units=[], return_unit=1,
                          codes=codes)


class ClipFunction(Function):
    '''
    A specifier for the clip function, allowing its use both in Python and
    C++ code. Only works for scalar ``a_min`` and ``a_max`` arguments or other
    state variables.

    '''
    def __init__(self):
        codes = {'python': lambda array, a_min, a_max: np.clip(array, a_min, a_max),
                 'cpp': {
                     'support_code':
        '''
        double _clip(const float value, const float a_min, const float a_max)
        {
	        if (value < a_min)
	            return a_min;
	        if (value > a_max)
	            return a_max;
	        return value;
	    }
        '''
                 }}
        Function.__init__(self, pyfunc=np.clip, arg_units=[None, None, None],
                          return_unit=lambda u1, u2, u3: u1,
                          codes=codes)


class IntFunction(Function):
    '''
    An ``int`` function for converting a boolean value into an integer.
    '''
    def __init__(self):
        codes = {'python': lambda value: np.int_(value),
                 'cpp': {'support_code':
        '''
        int int_(const bool value)
        {
	        return value ? 1 : 0;
        }
        '''
                 }}
        Function.__init__(self, pyfunc=np.int_, arg_units=[1],
                          return_unit=1, codes=codes)


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
    arg_units : list of `Unit`, optional
        The expected units of the arguments, ``None`` for arguments that can
        have arbitrary units. Needs only to be specified if the `pyfunc`
        function does not specify this already (e.g. via a `check_units`
        decorator)
    return_unit : `Unit` or callable, optional
        The unit of the return value of this function. Either a fixed `Unit`,
        or a function of the units of the arguments, e.g.
        ``lambda u : u **0.5`` for a square root function. Needs only to be
        specified if the `pyfunc` function does not specify this already (e.g.
        via a `check_units` decorator)
    '''
    # TODO: How to make this easily extendable for other languages?
    def __init__(self, pyfunc, py_name=None, cpp_name=None, sympy_func=None,
                 arg_units=None, return_unit=None):
        if py_name is None:
            py_name = pyfunc.__name__
        self.py_name = py_name
        self.cpp_name = cpp_name
        if self.cpp_name is None:
            hashdefine_code = ''
        else:
            hashdefine_code = '#define {python_name} {cpp_name}'.format(python_name=self.py_name,
                                                                        cpp_name=self.cpp_name)
        codes = {'python': lambda *args: pyfunc(*args),
                 'cpp': {'hashdefine_code': hashdefine_code}
                }
        Function.__init__(self, pyfunc, sympy_func, arg_units=arg_units,
                          return_unit=return_unit, codes=codes)

# sympy does not have a log10 function, so let's define one
class log10(sympy_Function):
    nargs = 1

    @classmethod
    def eval(cls, args):
        return sympy.functions.elementary.exponential.log(args, 10)


def _get_default_functions():

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
                                        sympy_func=sympy.functions.elementary.miscellaneous.sqrt,
                                        arg_units=[None], return_unit=lambda u: u**0.5),
                'ceil': FunctionWrapper(np.ceil,
                                        sympy_func=sympy.functions.elementary.integers.ceiling,
                                        arg_units=[None], return_unit=lambda u: u),
                'floor': FunctionWrapper(np.floor,
                                         sympy_func=sympy.functions.elementary.integers.floor,
                                         arg_units=[None], return_unit=lambda u: u),
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
                'abs': FunctionWrapper(np.abs, py_name='abs',
                                       cpp_name='fabs',
                                       sympy_func=sympy.functions.elementary.complexes.Abs,
                                       arg_units=[None], return_unit=lambda u: u),
                'mod': FunctionWrapper(np.mod, py_name='mod',
                                       cpp_name='fmod',
                                       sympy_func=sympy_mod.Mod,
                                       arg_units=[None, None], return_unit=lambda u,v : u),
                # functions that need special treatment
                'rand': RandFunction(),
                'randn': RandnFunction(),
                'clip': ClipFunction(),
                'int_': IntFunction()
                }

    return functions

DEFAULT_FUNCTIONS = _get_default_functions()