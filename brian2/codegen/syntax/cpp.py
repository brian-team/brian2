'''
Convert from Python syntax to C++ syntax

TODO: handle constants properly
TODO: handle pow/fpow abs/fabs correctly
'''

from brian2.utils.stringtools import word_substitute
from brian2.codegen.syntax.parser import (Var, PowVar, FuncVar,
                                          parse_expr, parse_statement,
                                          AndVar, OrVar, XorVar, Statement,
                                          )

__all__ = ['cpp_expr', 'cpp_statement']

cpp_function_substitutions = {
    }

def replace(var):
    newitems = []
    if isinstance(var, PowVar):
        var = FuncVar(Var('pow'), *var.items)
    elif isinstance(var, AndVar):
        var.op = '&&'
    elif isinstance(var, OrVar):
        var.op = '||'
    elif isinstance(var, XorVar):
        var.op = '^^'
    else:
        var.items = [replace(item) for item in var.items]
    return var

def cpp_statement(expr, use_float_math=False):
    var = parse_statement(expr)
    expr = replace(var.expr)
    op = var.op
    if op=='&=':
        op = '&&='
    elif op=='|=':
        op = ' ||='
    elif op=='^=':
        op = '^^='
    elif op=='**=':
        expr = FuncVar('pow', Var(var.var), expr)
        op = '='
    var = Statement(var.var, op, expr)
    returnval = str(var)+';'
    subs = cpp_function_substitutions.copy()
    return word_substitute(returnval, cpp_function_substitutions)


def cpp_expr(expr, use_float_math=False):
    var = parse_expr(expr)
    returnval = str(replace(var))
    subs = cpp_function_substitutions.copy()
    return word_substitute(returnval, cpp_function_substitutions)


if __name__=='__main__':
    expr = 'a+b*c+d(e)+f**g+(d&c)+(a<b+d)+(x==y)+(u>=v)'
    print cpp_expr(expr)
    print cpp_statement('a += 3*a')
    print cpp_statement('a = 0')
    print cpp_statement('a **= 5')
