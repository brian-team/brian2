from symbolic import symbolic_eval

__all__ = ['Expression']

class Expression(object):
    def __init__(self, expr):
        self.expr = expr
        self.sympy_expr = symbolic_eval(self.expr)
    
    def __str__(self):
        return self.expr
    __repr__ = __str__
    
    def translate_to(self, language):
        return language.translate_expression(self)
