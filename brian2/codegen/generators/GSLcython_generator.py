from .cython_generator import *
from .base import CodeGenerator
from brian2.core.variables import AuxiliaryVariable

__all__ = ['GSLCythonCodeGenerator']

class GSLCythonCodeGenerator(CythonCodeGenerator):

    lio_dic = {}

    def translate_statement(self, statement):
        code = CythonCodeGenerator.translate_statement(self, statement)
        var, op, expr, comment = (statement.var, statement.op,
                                  statement.expr, statement.comment)
        if '_lio' in var:
            self.lio_dic[var] = AuxiliaryVariable(var, dtype=statement.dtype)
        return code

    def translate(self, *args, **kwargs):
        scalar_code, vector_code, kwds = CodeGenerator.translate(self, *args, **kwargs)
        kwds['lio_dic'] = self.lio_dic
        return scalar_code, vector_code, kwds

