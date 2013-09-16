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


def add_implementations(function, codes, namespaces=None, name=None):
    if namespaces is None:
        namespaces = {}
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
        function.implementations[target_obj] = FunctionImplementation(name=name,
                                                                      code=code,
                                                                      namespace=namespace)