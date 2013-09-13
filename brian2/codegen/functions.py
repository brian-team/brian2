from brian2.core.functions import Function, FunctionImplementation
from .targets import codegen_targets

__all__ = ['make_function']


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
            function.implementations[target_obj] = FunctionImplementation(func.__name__,
                                                                          code=code)
        return function
    return do_make_user_function