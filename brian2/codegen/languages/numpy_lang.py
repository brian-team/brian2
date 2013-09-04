import numpy as np

from brian2.utils.stringtools import word_substitute
from brian2.parsing.rendering import NumpyNodeRenderer
from brian2.core.functions import (DEFAULT_FUNCTIONS, Function,
                                   FunctionImplementation)

from .base import Language

__all__ = ['NumpyLanguage']


class NumpyLanguage(Language):
    '''
    Numpy language
    
    Essentially Python but vectorised.
    '''

    language_id = 'numpy'

    def translate_expression(self, expr, namespace):
        for varname, var in namespace.iteritems():
            if isinstance(var, Function):
                impl_name = var.implementations[self.language_id].name
                if varname != impl_name:
                    expr = word_substitute(expr, {varname: impl_name})
        return NumpyNodeRenderer().render_expr(expr, namespace).strip()

    def translate_statement(self, statement, namespace):
        # TODO: optimisation, translate arithmetic to a sequence of inplace
        # operations like a=b+c -> add(b, c, a)
        var, op, expr = statement.var, statement.op, statement.expr
        if op == ':=':
            op = '='
        return var + ' ' + op + ' ' + self.translate_expression(expr, namespace)

    def translate_statement_sequence(self, statements, variables, namespace,
                                     variable_indices, iterate_all):
        read, write = self.array_read_write(statements, variables)
        lines = []
        # read arrays
        for var in read:
            spec = variables[var]
            index = variable_indices[var]
            line = var + ' = ' + spec.arrayname
            if not index in iterate_all:
                line = line + '[' + index + ']'
            lines.append(line)
        # the actual code
        lines.extend([self.translate_statement(stmt, namespace)
                      for stmt in statements])
        # write arrays
        for var in write:
            index_var = variable_indices[var]
            # check if all operations were inplace and we're operating on the
            # whole vector, if so we don't need to write the array back
            if not index_var in iterate_all:
                all_inplace = False
            else:
                all_inplace = True
                for stmt in statements:
                    if stmt.var == var and not stmt.inplace:
                        all_inplace = False
                        break
            if not all_inplace:
                line = variables[var].arrayname
                if index_var in iterate_all:
                    line = line + '[:]'
                else:
                    line = line + '[' + index_var + ']'
                line = line + ' = ' + var
                lines.append(line)
        return lines, {}

################################################################################
# Implement functions
################################################################################
# Functions that exist under the same name in numpy
for func in ['sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'exp', 'log',
             'log10', 'sqrt', 'ceil', 'floor', 'arcsin', 'arccos', 'arctan',
             'abs', 'mod']:
    DEFAULT_FUNCTIONS[func].implementations['numpy'] = FunctionImplementation(func)

# Functions that are implemented in a somewhat special way
randn_func = lambda vectorisation_idx: np.random.randn(len(vectorisation_idx))
DEFAULT_FUNCTIONS['randn'].implementations['numpy'] = FunctionImplementation('randn',
                                                                              code=randn_func)
rand_func = lambda vectorisation_idx: np.random.rand(len(vectorisation_idx))
DEFAULT_FUNCTIONS['rand'].implementations['numpy'] = FunctionImplementation('rand',
                                                                             code=rand_func)
clip_func = lambda array, a_min, a_max: np.clip(array, a_min, a_max)
DEFAULT_FUNCTIONS['clip'].implementations['numpy'] = FunctionImplementation('clip',
                                                                             code=clip_func)
int_func = lambda value: np.int_(value)
DEFAULT_FUNCTIONS['int_'].implementations['numpy'] = FunctionImplementation('int_',
                                                                             code=int_func)
