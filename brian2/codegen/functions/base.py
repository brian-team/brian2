__all__ = ['Function', 'SimpleFunction', 'make_function']

class Function(object):    
    def __init__(self, pyfunc, sympy_func=None):
        self.pyfunc = pyfunc
        self.sympy_func = sympy_func
    
    '''
    User-defined function to work with code generation
    
    To define a language, the user writes two methods,
    ``code_id(self, language, var)`` and
    ``on_compile_id(self, language, var, namespace)``, where ``id`` is replaced
    by the ``Language.language_id`` attribute of the target language. See below
    for the arguments and return values of these methods. Essentially, the
    idea is that ``on_compile`` should insert data needed into the namespace,
    and ``code`` should return code in the target language.
    
    By default, the Python language is implemented simply by inserting the
    object into the namespace, which will work if the class has a
    ``__call__`` method with the appropriate arguments.
    '''
    def code(self, language, var):
        """
        Returns a dict of ``(slot, section)`` values, where ``slot`` is a
        language-specific slot for where to include the string ``section``. The
        input arguments are the language object, and the variable name.
        Generated code should use unique identifiers of the form
        ``_func_var`` (where ``var`` is replaced by the value of ``var``) to
        ensure there are no namespace clashes.
        
        The list of slot names to use is language-specific.
        """
        try:
            return getattr(self, 'code_'+language.language_id)(language, var)
        except AttributeError:
            raise NotImplementedError
        
    def on_compile(self, namespace, language, var):
        """
        What to do at compile time, i.e. insert values into a namespace.
        """
        try:
            return getattr(self, 'on_compile_'+language.language_id)(namespace,
                                                                     language,
                                                                     var)
        except AttributeError:
            raise NotImplementedError

    # default implementation for Python is just to use the object itself,
    # which assumes it has a __call__ method
    def code_python(self, language, var):
        if not hasattr(self, '__call__'):
            return NotImplementedError
        return {}
    
    def on_compile_python(self, namespace, language, var):
        namespace[var] = self

    def __call__(self, *args):
        '''
        A `Function` specifier is callable, it applies the `Function.pyfunc`
        function to the arguments. This way, unit checking works.
        '''
        return self.pyfunc(*args)


class SimpleFunction(Function):
    '''
    A simplified, less generic version of `UserFunction`.
    
    You provide a dict ``codes`` of ``(language_id, code)`` pairs and a
    namespace of values to be added to the generated code. The ``code`` should
    be in the format recognised by the language (e.g. dict or string). In
    addition, you can specify a Python function ``pyfunc`` to call directly if
    the language is Python.
    '''
    def __init__(self, codes, namespace, pyfunc):
        Function.__init__(self, pyfunc)
        self.codes = codes
        self.namespace = namespace
        
    def code(self, language, var):
        if language.language_id in self.codes:
            return self.codes[language.language_id]
        else:
            raise NotImplementedError
        
    def on_compile(self, namespace, language, var):
        namespace.update(self.namespace)
        if language.language_id=='python':
            namespace[var] = self.pyfunc
        

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
        return SimpleFunction(codes, namespace, pyfunc=func)
    return do_make_user_function
