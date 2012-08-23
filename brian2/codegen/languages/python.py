from base import Language

__all__ = ['PythonLanguage']

class PythonLanguage(Language):
    
    def translate_expression(self, expr):
        return expr.strip()
    
    def translate_statement(self, statement):
        var, op, expr = statement.var, statement.op, statement.expr
        if op==':=':
            op = '='
        return var+' '+op+' '+self.translate_expression(expr)

    def translate_statement_sequence(self, statements, specifiers,
                                     index_var, index_spec):
        read, write = self.array_read_write(statements, specifiers)
        lines = []
        # read arrays
        for var in read:
            line = var+' = '+specifiers[var].array
            if not index_spec.all:
                line = line+'['+index_var+']'
            lines.append(line)
        # the actual code
        lines.extend([self.translate_statement(stmt) for stmt in statements])
        # write arrays
        # TODO: optimisation, if we have never done var = expr but always
        # var += expr or things like that, AND if we are iterating over the
        # whole array, then we do not need to do this final write of the
        # variable, and this is a common use-case in state update code.
        for var in write:
            line = specifiers[var].array
            if index_spec.all:
                line = line+'[:]'
            else:
                line = line+'['+index_var+']'
            line = line+' = '+var
            lines.append(line)
        return '\n'.join(lines)
