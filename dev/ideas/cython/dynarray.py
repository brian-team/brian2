from brian2 import *
from brian2.memory.dynamicarray import *
from cython import inline
from brian2.codegen.runtime.cython_rt.modified_inline import modified_cython_inline as inline

x = DynamicArray1D(10)

code = '''
print 'hi'
for i in xrange(len(x)):
    print x[i]
'''

ns = {'x': x}

a, b = inline(code, locals=ns, globals={})
print dir(a)
print b
a.__invoke(*b)