from brian2.utils.stringtools import get_identifiers
import sympy

__all__ = ['symbolic_eval']

def symbolic_eval(expr):
    """
    Evaluates expr as a sympy symbolic expression.
    """
    # TODO: not with all symbols
    # Find all symbols
    namespace = {}
    vars = get_identifiers(expr)
    for var in vars:
        namespace[var] = sympy.Symbol(var)
    return sympy.sympify(eval(expr, namespace))

if __name__=='__main__':
    expr = symbolic_eval('x+y*f(z)')
    print expr
    print expr.__class__
