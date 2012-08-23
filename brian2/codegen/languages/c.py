import numpy
from base import Language
from brian2.codegen.symbolic import symbolic_eval
from sympy.printing.ccode import CCodePrinter

__all__ = ['CLanguage',
           'c_data_type',
           ]

def c_data_type(dtype):
    '''
    Gives the C language specifier for numpy data types. For example,
    ``numpy.int32`` maps to ``int32_t`` in C.
    '''
    # this handles the case where int is specified, it will be int32 or int64
    # depending on platform
    if dtype is int:
        dtype = array([1]).dtype.type
        
    if dtype==numpy.float32:
        dtype = 'float'
    elif dtype==numpy.float64:
        dtype = 'double'
    elif dtype==numpy.int32:
        dtype = 'int32_t'
    elif dtype==numpy.int64:
        dtype = 'int64_t'
    elif dtype==numpy.uint16:
        dtype = 'uint16_t'
    elif dtype==numpy.uint32:
        dtype = 'uint32_t'
    elif dtype==numpy.bool_ or dtype is bool:
        dtype = 'bool'
    else:
        raise ValueError("dtype "+str(dtype)+" not known.")
    return dtype

class CLanguage(Language):
    def translate_expression(self, expr):
        expr = symbolic_eval(expr)
        return CCodePrinter().doprint(expr)

    def translate_statement(self, statement):
        var, op, expr = statement.var, statement.op, statement.expr
        if op==':=':
            decl = c_data_type(statement.dtype)+' '
            op = '='
            if statement.constant:
                decl = 'const '+decl
        else:
            decl = ''
        return decl+var+' '+op+' '+self.translate_expression(expr)+';'

    def translate_statement_sequence(self, statements, specifiers,
                                     index_var, index_spec):
        read, write = self.array_read_write(statements, specifiers)
        lines = []
        # read arrays
        for var in read:
            spec = specifiers[var]
            if var not in write:
                line = 'const '
            else:
                line = ''
            line = line+c_data_type(spec.dtype)+' '+var+' = '
            line = line+spec.array+'['+index_var+'];'
            lines.append(line)
        # the actual code
        lines.extend([self.translate_statement(stmt) for stmt in statements])
        # write arrays
        for var in write:
            spec = specifiers[var]
            line = spec.array+'['+index_var+'] = '+var+';'
            lines.append(line)
        return '\n'.join(lines)
