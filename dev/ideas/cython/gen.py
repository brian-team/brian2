from pylab import *
from brian2 import *
from brian2.codegen.generators.cython_generator import CythonCodeGenerator
from brian2.codegen.statements import Statement
from brian2.codegen.translation import make_statements
from brian2.core.variables import Variable, ArrayVariable

owner = None

code = '''
x = a*b+c
'''

variables = {'a':ArrayVariable('a', Unit(1), owner, 10, get_device()),
             'b':ArrayVariable('b', Unit(1), owner, 10, get_device()),
             'x':ArrayVariable('x', Unit(1), owner, 10, get_device()),
             }
namespace = {}
variable_indices = {'a': '_idx', 'b': '_idx', 'x': '_idx'}

gen = CythonCodeGenerator(variables, variable_indices, owner, iterate_all=True, codeobj_class=None)

#print gen.translate_expression('a*b+c')
#print gen.translate_statement(Statement('x', '=', 'a*b+c', '', float))

stmts = make_statements(code, variables, float)
#for stmt in stmts:
# print stmt

print '\n'.join(gen.translate_one_statement_sequence(stmts))
