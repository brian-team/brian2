from numpy.random import randn

from .base import Function

__all__ = ['RandnFunction']

class RandnFunction(Function):
    
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
    
