import itertools

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

    def translate_expression(self, expr, namespace, codeobj_class):
        for varname, var in namespace.iteritems():
            if isinstance(var, Function):
                impl_name = var.implementations[codeobj_class].name
                if impl_name is not None:
                    expr = word_substitute(expr, {varname: impl_name})
        return NumpyNodeRenderer().render_expr(expr, namespace).strip()

    def translate_statement(self, statement, namespace, codeobj_class):
        # TODO: optimisation, translate arithmetic to a sequence of inplace
        # operations like a=b+c -> add(b, c, a)
        var, op, expr = statement.var, statement.op, statement.expr
        if op == ':=':
            op = '='
        return var + ' ' + op + ' ' + self.translate_expression(expr,
                                                                namespace,
                                                                codeobj_class)

    def translate_one_statement_sequence(self, statements, variables, namespace,
                                         variable_indices, iterate_all,
                                         codeobj_class):
        read, write, indices = self.array_read_write(statements, variables,
                                            variable_indices)
        lines = []
        # index and read arrays (index arrays first)
        for var in itertools.chain(indices, read):
            spec = variables[var]
            index = variable_indices[var]
            line = var + ' = ' + spec.arrayname
            if not index in iterate_all:
                line = line + '[' + index + ']'
            lines.append(line)
        # the actual code
        lines.extend([self.translate_statement(stmt, namespace, codeobj_class)
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

        # Make sure we do not use the __call__ function of Function objects but
        # rather the Python function stored internally. The __call__ function
        # would otherwise return values with units
        for varname, var in namespace.iteritems():
            if isinstance(var, Function):
                namespace[varname] = var.implementations[codeobj_class].code

        return lines

    def translate_statement_sequence(self, statements, variables, namespace,
                                     variable_indices, iterate_all,
                                     codeobj_class):
        if isinstance(statements, dict):
            blocks = {}
            all_kwds = {}
            for name, block in statements.iteritems():
                blocks[name] = self.translate_one_statement_sequence(block,
                                                                     variables,
                                                                     namespace,
                                                                     variable_indices,
                                                                     iterate_all,
                                                                     codeobj_class)
            return blocks, {}
        else:
            block = self.translate_one_statement_sequence(statements, variables,
                                                          namespace, variable_indices,
                                                          iterate_all, codeobj_class)
            return block, {}

################################################################################
# Implement functions
################################################################################
# Functions that exist under the same name in numpy
for func_name, func in [('sin', np.sin), ('cos', np.cos), ('tan', np.tan),
                        ('sinh', np.sinh), ('cosh', np.cosh), ('tanh', np.tanh),
                        ('exp', np.exp), ('log', np.log), ('log10', np.log10),
                        ('sqrt', np.sqrt), ('ceil', np.ceil),
                        ('floor', np.floor), ('arcsin', np.arcsin),
                        ('arccos', np.arccos), ('arctan', np.arctan),
                        ('abs', np.abs), ('mod', np.mod)]:
    DEFAULT_FUNCTIONS[func_name].implementations[NumpyLanguage] = FunctionImplementation(code=func)

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
DEFAULT_FUNCTIONS['randn'].implementations[NumpyLanguage] = FunctionImplementation(code=randn_func)
DEFAULT_FUNCTIONS['rand'].implementations[NumpyLanguage] = FunctionImplementation(code=rand_func)
clip_func = lambda array, a_min, a_max: np.clip(array, a_min, a_max)
DEFAULT_FUNCTIONS['clip'].implementations[NumpyLanguage] = FunctionImplementation(code=clip_func)
int_func = lambda value: np.int_(value)
DEFAULT_FUNCTIONS['int_'].implementations[NumpyLanguage] = FunctionImplementation(code=int_func)
