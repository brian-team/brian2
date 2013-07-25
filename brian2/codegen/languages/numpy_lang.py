import os

from .base import Language
from brian2.parsing.rendering import NumpyNodeRenderer

__all__ = ['NumpyLanguage']


class NumpyLanguage(Language):
    '''
    Numpy language
    
    Essentially Python but vectorised.
    '''

    language_id = 'numpy'

    def translate_expression(self, expr):
        return NumpyNodeRenderer().render_expr(expr).strip()

    def translate_statement(self, statement):
        # TODO: optimisation, translate arithmetic to a sequence of inplace
        # operations like a=b+c -> add(b, c, a)
        var, op, expr = statement.var, statement.op, statement.expr
        if op == ':=':
            op = '='
        return var + ' ' + op + ' ' + self.translate_expression(expr)

    def translate_statement_sequence(self, statements, specifiers, namespace,
                                     iterate_all):
        read, write = self.array_read_write(statements, specifiers)
        lines = []
        # read arrays
        for var in read:
            spec = specifiers[var]
            line = var + ' = ' + spec.arrayname
            if not spec.index in iterate_all:
                line = line + '[' + spec.index + '_idx]'
            lines.append(line)
        # the actual code
        lines.extend([self.translate_statement(stmt) for stmt in statements])
        # write arrays
        for var in write:
            index_var = specifiers[var].index
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
                line = specifiers[var].arrayname
                if index_var in iterate_all:
                    line = line + '[:]'
                else:
                    line = line + '[' + index_var + '_idx]'
                line = line + ' = ' + var
                lines.append(line)
        return lines, {}
