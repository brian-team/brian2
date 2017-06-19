from .cython_generator import *
from brian2.parsing.bast import brian_dtype_from_dtype
from brian2.core.variables import AuxiliaryVariable

__all__ = ['GSLCythonCodeGenerator']

class GSLCythonCodeGenerator(CythonCodeGenerator):

    def translate_statement(self, statement):
        var, op, expr, comment = (statement.var, statement.op,
                                  statement.expr, statement.comment)
        if op == ':=': # make no distinction in Cython (declaration are done elsewhere)
            if var not in self.variables:
                self.variables[var] = AuxiliaryVariable(var, dtype=statement.dtype) #TODO: dimensions?
            op = '='
        # For Cython we replace complex expressions involving boolean variables into a sequence of
        # if/then expressions with simpler expressions. This is provided by the optimise_statements
        # function.
        if (statement.used_boolean_variables is not None and len(statement.used_boolean_variables)
                # todo: improve dtype analysis so that this isn't necessary
                and brian_dtype_from_dtype(statement.dtype)=='float'):
            used_boolvars = statement.used_boolean_variables
            bool_simp = statement.boolean_simplified_expressions
            codelines = []
            firstline = True
            # bool assigns is a sequence of (var, value) pairs giving the conditions under
            # which the simplified expression simp_expr holds
            for bool_assigns, simp_expr in bool_simp.iteritems():
                # generate a boolean expression like ``var1 and var2 and not var3``
                atomics = []
                for boolvar, boolval in bool_assigns:
                    if boolval:
                        atomics.append(boolvar)
                    else:
                        atomics.append('not '+boolvar)
                # use if/else/elif correctly
                if firstline:
                    line = 'if '+(' and '.join(atomics))+':'
                else:
                    if len(used_boolvars)>1:
                        line = 'elif '+(' and '.join(atomics))+':'
                    else:
                        line = 'else:'
                line += '\n    '
                line += var + ' ' + op + ' ' + self.translate_expression(simp_expr)
                codelines.append(line)
                firstline = False
            code = '\n'.join(codelines)
        else:
            code = var + ' ' + op + ' ' + self.translate_expression(expr)
        if len(comment):
            code += ' # ' + comment
        return code
