import os

import numpy as np

from ..base import Language, CodeObject
from ..templates import LanguageTemplater
from brian2.parsing.rendering import NumpyNodeRenderer

__all__ = ['PythonLanguage', 'PythonCodeObject']


class PythonLanguage(Language):

    language_id = 'python'

    templater = LanguageTemplater(os.path.join(os.path.split(__file__)[0],
                                               'templates'))

    def translate_expression(self, expr):
        return NumpyNodeRenderer().render_expr(expr).strip()

    def translate_statement(self, statement):
        # TODO: optimisation, translate arithmetic to a sequence of inplace
        # operations like a=b+c -> add(b, c, a)
        var, op, expr = statement.var, statement.op, statement.expr
        if op == ':=':
            op = '='
        return var + ' ' + op + ' ' + self.translate_expression(expr)

    def translate_statement_sequence(self, statements, specifiers, namespace, indices):
        read, write = self.array_read_write(statements, specifiers)
        lines = []
        # read arrays
        for var in read:
            spec = specifiers[var]
            index_spec = indices[spec.index]
            line = var + ' = ' + spec.arrayname
            if not index_spec.iterate_all:
                line = line + '[' + spec.index + ']'
            lines.append(line)
        # the actual code
        lines.extend([self.translate_statement(stmt) for stmt in statements])
        # write arrays
        for var in write:
            index_var = specifiers[var].index
            index_spec = indices[index_var]
            # check if all operations were inplace and we're operating on the
            # whole vector, if so we don't need to write the array back
            if not index_spec.iterate_all:
                all_inplace = False
            else:
                all_inplace = True
                for stmt in statements:
                    if stmt.var == var and not stmt.inplace:
                        all_inplace = False
                        break
            if not all_inplace:
                line = specifiers[var].arrayname
                if index_spec.iterate_all:
                    line = line + '[:]'
                else:
                    line = line + '[' + index_var + ']'
                line = line + ' = ' + var
                lines.append(line)
        return lines, {}

    def code_object(self, code, namespace, specifiers):
        # TODO: This should maybe go somewhere else
        namespace['logical_not'] = np.logical_not
        return PythonCodeObject(code, namespace, specifiers,
                                self.compile_methods(namespace))


class PythonCodeObject(CodeObject):
    def compile(self):
        super(PythonCodeObject, self).compile()
        self.compiled_code = compile(self.code, '(string)', 'exec')

    def run(self):
        exec self.compiled_code in self.namespace
        # output variables should land in the variable name _return_values
        if '_return_values' in self.namespace:
            return self.namespace['_return_values']
