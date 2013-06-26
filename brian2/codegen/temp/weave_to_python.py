from numpy import *
from scipy import weave

class X(object):
    def __init__(self, N, p, data):
        self.N = N
        self.p = p
        self.data = data
    def getN(self):
        return self.N

code = r'''
int N = x.mcall("getN");
printf("%d\n", N);
printf("%g\n", (double)(x.attr("p")));
double *data = (double*)(((PyArrayObject*)(PyObject*)x.attr("data"))->data);
data[0] = 3.5;
//py::array data = x.attr("data");
//x.attr("data").test();
'''

x = X(3, 0.1, zeros(3))
ns = {'x': x,
      }

weave.inline(code, ns.keys(), ns, compiler='gcc')

print x.data
