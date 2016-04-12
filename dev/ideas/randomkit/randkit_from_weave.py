from brian2 import *
from scipy import weave
import brian2
import os

# code = '''
# int x;
#
# npy_intp dims[1] = {1};
#
# PyObject* out_array = PyArray_SimpleNew(1, dims, NPY_INTP);
# ((npy_intp*) ((PyArrayObject*) out_array)->data)[0] = (npy_intp)(&x);
#
# return_val = out_array;
# Py_XDECREF(out_array);
# '''
# print weave.inline(code, [], {}, compiler='msvc')
#
# exit()

brian2dir, _ = os.path.split(brian2.__file__)
rkdir = os.path.join(brian2dir, 'random', 'randomkit')
rkc = os.path.join(rkdir, 'rk.cpp')
randomkitc = os.path.join(rkdir, 'randomkit.c')

code = '''
rk_state **internal_state = get_rk_state();
if(*internal_state==NULL)
{
    *internal_state = new rk_state;
    rk_error errcode = rk_randomseed(*internal_state);
    if(errcode)
    {
        PyErr_SetString(PyExc_RuntimeError, "Cannot initialise random state");
        throw 1;
    }
    std::cout << "Allocated new random state." << std::endl;
}
std::cout << rk_double(*internal_state) << std::endl;
'''

code = '''
rk_state *internal_state = NULL;
internal_state = new rk_state;
rk_error errcode = rk_randomseed(internal_state);
if(errcode)
{
    PyErr_SetString(PyExc_RuntimeError, "Cannot initialise random state");
    throw 1;
}
std::cout << rk_double(internal_state) << std::endl;
delete internal_state;
'''

for i in range(100000):
    weave.inline(#code+'\n//'+str(i)+'\n',
                 code,
                 [], {}, compiler='msvc',
                 headers=['"randomkit.h"', '"rk.h"'], sources=[rkc, randomkitc],
                 libraries=['advapi32'], include_dirs=[rkdir])
