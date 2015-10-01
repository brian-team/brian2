from brian2 import *
from brian2.core.variables import *
from brian2.codegen.translation import apply_loop_invariant_optimisations

code = '''
x += y
x += y*y
x += exp(y*int(b))
'''

variables = dict(
    x=ArrayVariable('x', 1, None, 1, device),
    y=ArrayVariable('y', 1, None, 1, device, scalar=True),
    b=ArrayVariable('b', 1, None, 1, device, dtype=bool),
)

scalar_statements, vector_statements = make_statements(code, variables, float64)

print 'SCALAR'
for stmt in scalar_statements:
    print '   ', stmt
print 'VECTOR'
for stmt in vector_statements:
    print '   ', stmt