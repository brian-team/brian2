from .cython_generator import *
from .base import CodeGenerator
from brian2.core.variables import AuxiliaryVariable
from brian2.codegen.translation import make_statements

from brian2.codegen.permutation_analysis import (check_for_order_independence,
                                                 OrderDependenceError)

__all__ = ['GSLCodeGenerator']

class GSLCodeGenerator(object): #TODO: I don't think it matters it doesn't inherit from CodeGenerator (the base) because it can access this through __getattr__ of the parent anyway?
    def __init__(self, *args, **kwargs):
        self.other_variables = {}
        self.generator = CythonCodeGenerator(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self.generator, item)

    def translate(self, code, dtype): # TODO: it's not so nice we have to copy the contents of this function..
        '''
        Translates an abstract code block into the target language.
        '''
        scalar_statements = {}
        vector_statements = {}
        for ac_name, ac_code in code.iteritems():
            statements = make_statements(ac_code,
                                         self.variables,
                                         dtype,
                                         optimise=True,
                                         blockname=ac_name)
            scalar_statements[ac_name], vector_statements[ac_name] = statements
        for vs in vector_statements.itervalues():
            # Check that the statements are meaningful independent on the order of
            # execution (e.g. for synapses)
            try:
                if self.has_repeated_indices(vs):  # only do order dependence if there are repeated indices
                    check_for_order_independence(vs,
                                                 self.generator.variables,
                                                 self.generator.variable_indices)
            except OrderDependenceError:
                # If the abstract code is only one line, display it in full
                if len(vs) <= 1:
                    error_msg = 'Abstract code: "%s"\n' % vs[0]
                else:
                    error_msg = ('%d lines of abstract code, first line is: '
                                 '"%s"\n') % (len(vs), vs[0])

        scalar_code, vector_code, kwds = self.generator.translate_statement_sequence(scalar_statements,
                                                 vector_statements)

        for dictionary in [scalar_statements, vector_statements]:
            for key, value in dictionary.items():
                for statement in value:
                     var, op, expr, comment = (statement.var, statement.op,
                                              statement.expr, statement.comment)
                     if var not in self.variables:
                         self.other_variables[var] = AuxiliaryVariable(var, dtype=statement.dtype)

        kwds['other_variables'] = self.other_variables
        return scalar_code, vector_code, kwds
