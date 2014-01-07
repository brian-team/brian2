'''
Module providing support for multi-language functions.
'''

import types

import numpy as np

from brian2.units.fundamentalunits import (fail_for_dimension_mismatch,
                                           get_dimensions)
from brian2.units.fundamentalunits import Quantity
from brian2.core.preferences import brian_prefs
from brian2.core.functions import Function, FunctionImplementation

from .targets import codegen_targets

__all__ = ['make_function']


def make_function(codes=None, namespaces=None, discard_units=None):
    '''
    A simple decorator to extend user-written Python functions to work with code
    generation in other languages.

    Parameters
    ----------
    codes : dict-like, optional
        A mapping from `Language` or `CodeObject` class objects, or their
        corresponding names (e.g. `'numpy'` or `'weave'`) to codes for the
        target language. What kind of code the target language expectes is
        language-specific, e.g. C++ code has to be provided as a dictionary
        of code blocks.
    namespaces : dict-like, optional
        If provided, has to use the same keys as the `codes` argument and map
        it to a namespace dictionary (i.e. a mapping of names to values) that
        should be added to a `CodeObject` namespace when using this function.
    discard_units: bool, optional
        Numpy functions can internally make use of the unit system. However,
        during a simulation run, state variables are passed around as unitless
        values for efficiency. If `discard_units` is set to ``False``, input
        arguments will have units added to them so that the function can still
        use units internally (the units will be stripped away from the return
        value as well). Alternatively, if `discard_units` is set to ``True``,
        the function will receive unitless values as its input. The namespace
        of the function will be altered to make references to units (e.g.
        ``ms``) refer to the corresponding floating point values so that no
        unit mismatch errors are raised. Note that this system cannot work in
        all cases, e.g. it does not work with functions that internally imports
        values (e.g. does ``from brian2 import ms``) or access values with
        units indirectly (e.g. uses ``brian2.ms`` instead of ``ms``). If no
        value is given, defaults to the preference setting
        `codegen.runtime.numpy.discard_units`.

    Notes
    -----
    While it is in principle possible to provide a numpy implementation
    as an argument for this decorator, this is normally not necessary -- the
    numpy implementation should be provided in the decorated function.

    Examples
    --------
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
    if codes is None:
        codes = {}

    def do_make_user_function(func):
        function = Function(func)

        if discard_units:  # Add a numpy implementation that discards units
            add_numpy_implementation(function, wrapped_func=func,
                                     discard_units=discard_units)

        add_implementations(function, codes, namespaces)
        return function
    return do_make_user_function


def add_numpy_implementation(function, wrapped_func, discard_units=None):
    '''
    Add a numpy implementation to a `Function`.

    Parameters
    ----------
    function : `Function`
        The function description for which an implementation should be added.
    wrapped_func : callable
        The original function (that will be used for the numpy implementation)
    discard_units : bool, optional
        See `make_function`.
    '''
    # do the import here to avoid cyclical imports
    from .runtime.numpy_rt.numpy_rt import NumpyCodeObject

    if discard_units is None:
        discard_units = brian_prefs['codegen.runtime.numpy.discard_units']

    # Get the original function inside the check_units decorator
    if hasattr(wrapped_func,  '_orig_func'):
        orig_func = wrapped_func._orig_func
    else:
        orig_func = wrapped_func

    if discard_units:
        new_globals = dict(orig_func.func_globals)
        # strip away units in the function by changing its namespace
        for key, value in new_globals.iteritems():
            if isinstance(value, Quantity):
                new_globals[key] = np.asarray(value)
        unitless_func = types.FunctionType(orig_func.func_code, new_globals,
                                           orig_func.func_name,
                                           orig_func.func_defaults,
                                           orig_func.func_closure)
        function.implementations[NumpyCodeObject] = FunctionImplementation(name=None,
                                                                           code=unitless_func)
    else:
        def wrapper_function(*args):
            if not len(args) == len(function._arg_units):
                raise ValueError(('Function %s got %d arguments, '
                                  'expected %d') % (function.name, len(args),
                                                    len(function._arg_units)))
            new_args = [Quantity.with_dimensions(arg, get_dimensions(arg_unit))
                        for arg, arg_unit in zip(args, function._arg_units)]
            result = orig_func(*new_args)
            fail_for_dimension_mismatch(result, function._return_unit)
            return np.asarray(result)

        function.implementations[NumpyCodeObject] = FunctionImplementation(name=None,
                                                                           code=wrapper_function)


def add_implementations(function, codes, namespaces=None, names=None):
    '''
    Add implementations to a `Function`.

    Parameters
    ----------
    function : `Function`
        The function description for which implementations should be added.
    codes : dict-like
        See `make_function`
    namespace : dict-like, optional
        See `make_function`
    names : dict-like, optional
        The name of the function in the given target language, if it should
        be renamed. Has to use the same keys as the `codes` and `namespaces`
        dictionary.
    '''
    if namespaces is None:
        namespaces = {}
    if names is None:
        names = {}
    for target, code in codes.iteritems():
        # Try to find the CodeObject or Language class, corresponding to the
        # given string
        if isinstance(target, basestring):
            target_obj = None
            for codegen_target in codegen_targets:
                if codegen_target.class_name == target:
                    target_obj = codegen_target
                    break
                elif codegen_target.language.language_id == target:
                    target_obj = codegen_target.language.__class__
                    break
            if target_obj is None:
                raise ValueError('Unknown code generation target %s' % target)
        else:
            target_obj = target
        namespace = namespaces.get(target, None)
        name = names.get(target, None)
        function.implementations[target_obj] = FunctionImplementation(name=name,
                                                                      code=code,
                                                                      namespace=namespace)