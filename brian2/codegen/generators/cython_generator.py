import itertools

import numpy as np

from brian2.utils.stringtools import word_substitute
from brian2.parsing.rendering import NumpyNodeRenderer
from brian2.core.functions import DEFAULT_FUNCTIONS, Function
from brian2.core.variables import ArrayVariable

from .base import CodeGenerator

__all__ = ['CythonCodeGenerator']


class CythonCodeGenerator(CodeGenerator):
    '''
    Cython code generator
    '''

    class_name = 'cython'

    def translate_expression(self, expr):
        # numpy version
#        for varname, var in self.variables.iteritems():
#            if isinstance(var, Function):
#                impl_name = var.implementations[self.codeobj_class].name
#                if impl_name is not None:
#                    expr = word_substitute(expr, {varname: impl_name})
#        return NumpyNodeRenderer().render_expr(expr, self.variables).strip()
        return expr.strip()

    def translate_statement(self, statement):
        # TODO: optimisation, translate arithmetic to a sequence of inplace
        # operations like a=b+c -> add(b, c, a)
        var, op, expr, comment = (statement.var, statement.op,
                                  statement.expr, statement.comment)
        if op == ':=':
            op = '='
        code = var + ' ' + op + ' ' + self.translate_expression(expr)
        if len(comment):
            code += ' # ' + comment
        return code
        
    def translate_one_statement_sequence(self, statements):
        variables = self.variables
        variable_indices = self.variable_indices
        read, write, indices, conditional_write_vars = self.arrays_helper(statements)
        lines = []
        # index and read arrays (index arrays first)
        for varname in itertools.chain(indices, read):
            var = variables[varname]
            index = variable_indices[varname]
            line = '{varname} = {arrayname}[{index}]'.format(varname=varname, arrayname=self.get_array_name(var),
                                                             index=index)
            lines.append(line)
        # the actual code
        created_vars = set([])
        for stmt in statements:
            if stmt.op==':=':
                created_vars.add(stmt.var)
            line = self.translate_statement(stmt)
            if stmt.var in conditional_write_vars:
                subs = {}
                condvar = conditional_write_vars[stmt.var]
                lines.append('if %s:' % condvar)
                lines.append('    '+line)
            lines.append(line)
        # write arrays
        for varname in write:
            index_var = self.variable_indices[varname]
            var = self.variables[varname]
            line = self.get_array_name(var, self.variables) + '[' + index_var + '] = ' + varname
            lines.append(line)
            
        # TODO: this was in numpy, do we want it for cython too?
        # Make sure we do not use the __call__ function of Function objects but
        # rather the Python function stored internally. The __call__ function
        # would otherwise return values with units
        for varname, var in variables.iteritems():
            if isinstance(var, Function):
                variables[varname] = var.implementations[self.codeobj_class].get_code(self.owner)

        return lines

    def translate_statement_sequence(self, statements):
        # For numpy, no addiional keywords are provided to the template
        scalar_code = {}
        vector_code = {}
        for name, block in statements.iteritems():
            scalar_statements = [stmt for stmt in block if stmt.scalar]
            vector_statements = [stmt for stmt in block if not stmt.scalar]
            scalar_code[name] = self.translate_one_statement_sequence(scalar_statements)
            vector_code[name] = self.translate_one_statement_sequence(vector_statements)
        return scalar_code, vector_code, {}

###############################################################################
# Implement functions
################################################################################
# Functions that exist under the same name in numpy
for func_name, func in [('sin', np.sin), ('cos', np.cos), ('tan', np.tan),
                        ('sinh', np.sinh), ('cosh', np.cosh), ('tanh', np.tanh),
                        ('exp', np.exp), ('log', np.log), ('log10', np.log10),
                        ('sqrt', np.sqrt), ('arcsin', np.arcsin),
                        ('arccos', np.arccos), ('arctan', np.arctan),
                        ('abs', np.abs), ('mod', np.mod)]:
    DEFAULT_FUNCTIONS[func_name].implementations.add_implementation(CythonCodeGenerator,
                                                                    code=func)

# Functions that are implemented in a somewhat special way
def randn_func(vectorisation_idx):
    try:
        N = int(vectorisation_idx)
    except (TypeError, ValueError):
        N = len(vectorisation_idx)

    return np.random.randn(N)

def rand_func(vectorisation_idx):
    try:
        N = int(vectorisation_idx)
    except (TypeError, ValueError):
        N = len(vectorisation_idx)

    return np.random.rand(N)
DEFAULT_FUNCTIONS['randn'].implementations.add_implementation(CythonCodeGenerator,
                                                              code=randn_func)
DEFAULT_FUNCTIONS['rand'].implementations.add_implementation(CythonCodeGenerator,
                                                             code=rand_func)
clip_func = lambda array, a_min, a_max: np.clip(array, a_min, a_max)
DEFAULT_FUNCTIONS['clip'].implementations.add_implementation(CythonCodeGenerator,
                                                             code=clip_func)
int_func = lambda value: np.int32(value)
DEFAULT_FUNCTIONS['int'].implementations.add_implementation(CythonCodeGenerator,
                                                            code=int_func)
ceil_func = lambda value: np.int32(np.ceil(value))
DEFAULT_FUNCTIONS['ceil'].implementations.add_implementation(CythonCodeGenerator,
                                                            code=ceil_func)
floor_func = lambda value: np.int32(np.floor(value))
DEFAULT_FUNCTIONS['floor'].implementations.add_implementation(CythonCodeGenerator,
                                                            code=floor_func)
