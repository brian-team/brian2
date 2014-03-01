import itertools

import numpy as np

from brian2.utils.stringtools import word_substitute
from brian2.parsing.rendering import NumpyNodeRenderer
from brian2.core.functions import DEFAULT_FUNCTIONS, Function
from brian2.core.variables import ArrayVariable

from .base import CodeGenerator

__all__ = ['NumpyCodeGenerator']


class NumpyCodeGenerator(CodeGenerator):
    '''
    Numpy language
    
    Essentially Python but vectorised.
    '''

    class_name = 'numpy'

    def translate_expression(self, expr):
        for varname, var in self.variables.iteritems():
            if isinstance(var, Function):
                impl_name = var.implementations[self.codeobj_class].name
                if impl_name is not None:
                    expr = word_substitute(expr, {varname: impl_name})
        return NumpyNodeRenderer().render_expr(expr, self.variables).strip()

    def translate_statement(self, statement):
        # TODO: optimisation, translate arithmetic to a sequence of inplace
        # operations like a=b+c -> add(b, c, a)
        var, op, expr = statement.var, statement.op, statement.expr
        if op == ':=':
            op = '='
        return var + ' ' + op + ' ' + self.translate_expression(expr)
        
    def translate_one_statement_sequence(self, statements):
        variables = self.variables
        variable_indices = self.variable_indices
        read, write, indices, conditional_write_vars = self.arrays_helper(statements)
        lines = []
        # index and read arrays (index arrays first)
        for varname in itertools.chain(indices, read):
            var = variables[varname]
            index = variable_indices[varname]
#            if index in iterate_all:
#                line = '{varname} = {array_name}'
#            else:
#                line = '{varname} = {array_name}.take({index})'
#            line = line.format(varname=varname, array_name=self.get_array_name(var), index=index)
            line = varname + ' = ' + self.get_array_name(var)
            if not index in self.iterate_all:
                line = line + '[' + index + ']'
            lines.append(line)
        # the actual code
        created_vars = set([])
        for stmt in statements:
            if stmt.op==':=':
                created_vars.add(stmt.var)
            line = self.translate_statement(stmt)
            if stmt.var in conditional_write_vars:
                subs = {}
                index = conditional_write_vars[stmt.var]
                # we replace all var with var[index], but actually we use this repl_string first because
                # we don't want to end up with lines like x[not_refractory[not_refractory]] when
                # multiple substitution passes are invoked
                repl_string = '#$(@#&$@$*U#@)$@(#' # this string shouldn't occur anywhere I hope! :)
                for varname, var in variables.items():
                    if isinstance(var, ArrayVariable):
                        subs[varname] = varname+'['+repl_string+']'
                # all newly created vars are arrays and will need indexing
                for varname in created_vars:
                    subs[varname] = varname+'['+repl_string+']'
                line = word_substitute(line, subs)
                line = line.replace(repl_string, index)
            lines.append(line)
        # write arrays
        for varname in write:
            var = variables[varname]
            index_var = variable_indices[varname]
            # check if all operations were inplace and we're operating on the
            # whole vector, if so we don't need to write the array back
            if not index_var in self.iterate_all:
                all_inplace = False
            else:
                all_inplace = True
                for stmt in statements:
                    if stmt.var == varname and not stmt.inplace:
                        all_inplace = False
                        break
            if not all_inplace:
                line = self.get_array_name(var)
                if index_var in self.iterate_all:
                    line = line + '[:]'
                else:
                    line = line + '[' + index_var + ']'
                line = line + ' = ' + varname
                lines.append(line)
#                if index_var in iterate_all:
#                    line = '{array_name}[:] = {varname}'
#                else:
#                    line = '''
#if isinstance(_idx, slice):
#    {array_name}[:] = {varname}
#else:
#    {array_name}.put({index_var}, {varname})
#                    '''
#                    line = '\n'.join([l for l in line.split('\n') if l.strip()])
#                line = line.format(array_name=self.get_array_name(var), index_var=index_var, varname=varname)
#                if index_var in iterate_all:
#                    lines.append(line)
#                else:
#                    lines.extend(line.split('\n'))

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

################################################################################
# Implement functions
################################################################################
# Functions that exist under the same name in numpy
for func_name, func in [('sin', np.sin), ('cos', np.cos), ('tan', np.tan),
                        ('sinh', np.sinh), ('cosh', np.cosh), ('tanh', np.tanh),
                        ('exp', np.exp), ('log', np.log), ('log10', np.log10),
                        ('sqrt', np.sqrt), ('arcsin', np.arcsin),
                        ('arccos', np.arccos), ('arctan', np.arctan),
                        ('abs', np.abs), ('mod', np.mod)]:
    DEFAULT_FUNCTIONS[func_name].implementations.add_implementation(NumpyCodeGenerator,
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
DEFAULT_FUNCTIONS['randn'].implementations.add_implementation(NumpyCodeGenerator,
                                                              code=randn_func)
DEFAULT_FUNCTIONS['rand'].implementations.add_implementation(NumpyCodeGenerator,
                                                             code=rand_func)
clip_func = lambda array, a_min, a_max: np.clip(array, a_min, a_max)
DEFAULT_FUNCTIONS['clip'].implementations.add_implementation(NumpyCodeGenerator,
                                                             code=clip_func)
int_func = lambda value: np.int_(value)
DEFAULT_FUNCTIONS['int'].implementations.add_implementation(NumpyCodeGenerator,
                                                            code=int_func)
ceil_func = lambda value: np.int_(np.ceil(value))
DEFAULT_FUNCTIONS['ceil'].implementations.add_implementation(NumpyCodeGenerator,
                                                            code=ceil_func)
floor_func = lambda value: np.int_(np.floor(value))
DEFAULT_FUNCTIONS['floor'].implementations.add_implementation(NumpyCodeGenerator,
                                                            code=floor_func)
