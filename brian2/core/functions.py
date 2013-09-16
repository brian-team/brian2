import collections

import sympy
from sympy import Function as sympy_Function
from sympy.core import mod as sympy_mod
import numpy as np
from numpy.random import randn, rand

import brian2.units.unitsafefunctions as unitsafe

__all__ = ['DEFAULT_FUNCTIONS', 'Function', 'FunctionImplementation']


class Function(object):
    '''
    An abstract specification of a function that can be used as part of
    model equations, etc.

    Parameters
    ----------
    pyfunc : function
        A Python function that is represented by this `Function` object.
    sympy_func : `sympy.Function`, optional
        A corresponding sympy function (if any). Allows functions to be
        interpreted by sympy and potentially make simplifications. For example,
        ``sqrt(x**2)`` could be replaced by ``abs(x)``.
    arg_units : list of `Unit`, optional
        If `pyfunc` does not provide unit information (which typically means
        that it was not annotated with a `check_units` decorator), the
        units of the arguments have to specified explicitly using this
        parameter.
    return_unit : `Unit` or callable, optional
        Same as for `arg_units`: if `pyfunc` does not provide unit information,
        this information has to be provided explictly here. `return_unit` can
        either be a specific `Unit`, if the function always returns the same
        unit, or a function of the input units, e.g. a "square" function would
        return the square of its input units, i.e. `return_unit` could be
        specified as ``lambda u: u**2``.

    Notes
    -----
    If a function should be usable for code generation targets other than
    Python/numpy, implementations for these target languages have to be added
    using the `~brian2.codegen.functions.make_function` decorator or using the
    `~brian2.codegen.functions.add_implementations` function.
    '''
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

        #: Stores implementations for this function in a
        #: `FunctionImplementationContainer`
        self.implementations = FunctionImplementationContainer()

    def __call__(self, *args):
        return self.pyfunc(*args)


class FunctionImplementation(object):
    '''
    A simple container object for function implementations.

    Parameters
    ----------
    name : str, optional
        The name of the function in the target language. Should only be
        specified if the function has to be renamed for the target language.
    code : language-dependent, optional
        A language dependent argument specifying the implementation in the
        target language, e.g. a code string or a dictionary of code strings.
    namespace : dict-like, optional
        A dictionary of mappings from names to values that should be added
        to the namespace of a `CodeObject` using the function.
    '''
    def __init__(self, name=None, code=None, namespace=None):
        self.name = name
        self.code = code
        self.namespace = namespace


class FunctionImplementationContainer(collections.MutableMapping):
    '''
    Helper object to store implementations and give access in a dictionary-like
    fashion, using `Language` implementations as a fallback for `CodeObject`
    implementations.
    '''
    def __init__(self):
        self._implementations = dict()

    def __getitem__(self, key):
        fallback = None
        if hasattr(key, 'language'):
            fallback = key.language.__class__

        if key in self._implementations:
            return self._implementations[key]
        elif fallback in self._implementations:
            return self._implementations[fallback]
        else:
            raise KeyError(('No implementation available for {key}. '
                            'Available implementations: {keys}').format(key=key,
                                                                        keys=self._implementations.keys()))

    def __setitem__(self, key, value):
        self._implementations[key] = value

    def __delitem__(self, key):
        del self._implementations[key]

    def __len__(self):
        return len(self._implementations)

    def __iter__(self):
        return iter(self._implementations)


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