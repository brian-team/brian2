import sympy
from sympy import Function as sympy_Function
from sympy.core import mod as sympy_mod
import numpy as np
from numpy.random import randn, rand

import brian2.units.unitsafefunctions as unitsafe
from brian2.units.fundamentalunits import Quantity, Unit, get_dimensions

__all__ = ['DEFAULT_FUNCTIONS', 'Function', 'make_function',
           'FunctionImplementation']


class Function(object):
    def __init__(self, pyfunc, sympy_func=None, arg_units=None,
                 return_unit=None):
        self.pyfunc = pyfunc
        self.sympy_func = sympy_func
        self._arg_units = arg_units
        self._return_unit = return_unit
        if self._arg_units is None:
            if hasattr(pyfunc, '_arg_units'):
                self._arg_units = pyfunc._arg_units
            else:
                raise ValueError('The given Python function does not specify '
                                  'how it deals with units, need to specify '
                                  '"arg_units".')
        if self._return_unit is None:
            if hasattr(pyfunc, '_return_unit'):
                self._return_unit = pyfunc._return_unit
            else:
                raise ValueError(('The given Python function does not specify '
                                  'how it deals with units, need to specify '
                                  '"return_unit".'))

        # Provide the numpy implementation by default
        self.implementations = {'numpy': FunctionImplementation(name=None,
                                                                code=pyfunc)}

    def implementation(self, codeobj_class):
        implementation = self.implementations.get(codeobj_class, None)
        if implementation is None:
            implementation = self.implementations.get(codeobj_class.language.language_id,
                                                      None)

        if implementation is None:
            raise NotImplementedError(('Function is not implemented for '
                                       'class %s or language %s') % (codeobj_class.__name__,
                                                                     codeobj_class.language.language_id))

        return implementation

    def __call__(self, *args):
        if callable(self._return_unit):
            return_dim = get_dimensions(self._return_unit(*args))
        else:
            return_dim = get_dimensions(self._return_unit)
        return Quantity.with_dimensions(self.implementation('numpy').code(*args),
                                        return_dim)


class FunctionImplementation(object):

    def __init__(self, name=None, code=None, namespace=None):
        self.name = name
        self.code = code
        self.namespace = namespace


def make_function(codes):
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
            })
        def usersin(x):
            return sin(x)
    '''
    def do_make_user_function(func):
        function = Function(func)
        for language_id, code in codes.iteritems():
            function.implementations[language_id] = FunctionImplementation(func.__name__,
                                                                           code=code)
        return function
    return do_make_user_function


################################################################################
# Standard functions
################################################################################

# sympy does not have a log10 function, so let's define one
class log10(sympy_Function):
    nargs = 1

    @classmethod
    def eval(cls, args):
        return sympy.functions.elementary.exponential.log(args, 10)


def _get_default_functions():

    functions = {
                # numpy functions that have the same name in numpy and math.h
                'cos': Function(unitsafe.cos,
                                sympy_func=sympy.functions.elementary.trigonometric.cos),
                'sin': Function(unitsafe.sin,
                                sympy_func=sympy.functions.elementary.trigonometric.sin),
                'tan': Function(unitsafe.tan,
                                sympy_func=sympy.functions.elementary.trigonometric.tan),
                'cosh': Function(unitsafe.cosh,
                                 sympy_func=sympy.functions.elementary.hyperbolic.cosh),
                'sinh': Function(unitsafe.sinh,
                                 sympy_func=sympy.functions.elementary.hyperbolic.sinh),
                'tanh': Function(unitsafe.tanh,
                                 sympy_func=sympy.functions.elementary.hyperbolic.tanh),
                'exp': Function(unitsafe.exp,
                                sympy_func=sympy.functions.elementary.exponential.exp),
                'log': Function(unitsafe.log,
                                sympy_func=sympy.functions.elementary.exponential.log),
                'log10': Function(unitsafe.log10,
                                  sympy_func=log10),
                'sqrt': Function(np.sqrt,
                                 sympy_func=sympy.functions.elementary.miscellaneous.sqrt,
                                 arg_units=[None], return_unit=lambda u: u**0.5),
                'ceil': Function(np.ceil,
                                 sympy_func=sympy.functions.elementary.integers.ceiling,
                                 arg_units=[None], return_unit=lambda u: u),
                'floor': Function(np.floor,
                                  sympy_func=sympy.functions.elementary.integers.floor,
                                  arg_units=[None], return_unit=lambda u: u),
                # numpy functions that have a different name in numpy and math.h
                'arccos': Function(unitsafe.arccos,
                                   sympy_func=sympy.functions.elementary.trigonometric.acos),
                'arcsin': Function(unitsafe.arcsin,
                                   sympy_func=sympy.functions.elementary.trigonometric.asin),
                'arctan': Function(unitsafe.arctan,
                                   sympy_func=sympy.functions.elementary.trigonometric.atan),
                'abs': Function(np.abs,
                                sympy_func=sympy.functions.elementary.complexes.Abs,
                                arg_units=[None], return_unit=lambda u: u),
                'mod': Function(np.mod,
                                sympy_func=sympy_mod.Mod,
                                arg_units=[None, None], return_unit=lambda u,v : u),
                # functions that need special treatment
                'rand': Function(pyfunc=rand, arg_units=[], return_unit=1),
                'randn': Function(pyfunc=randn, arg_units=[], return_unit=1),
                'clip': Function(pyfunc=np.clip, arg_units=[None, None, None],
                                 return_unit=lambda u1, u2, u3: u1,),
                'int_': Function(pyfunc=np.int_, arg_units=[1], return_unit=1)
                }

    return functions

DEFAULT_FUNCTIONS = _get_default_functions()