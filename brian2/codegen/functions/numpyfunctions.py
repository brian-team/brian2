from numpy.random import randn

from .base import Function

__all__ = ['RandnFunction', 'FunctionWrapper']

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
    '''
    # TODO: How to make this easily extendable for other languages?
    def __init__(self, pyfunc, py_name=None, cpp_name=None):
        Function.__init__(self, pyfunc)
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