from base import Language
from python import PythonLanguage
import sympy

__all__ = ['NumexprPythonLanguage']

class NumexprPythonLanguage(PythonLanguage):
    def __init__(self, complexity_threshold=2):
        '''
        ``complexity_threshold`` is the minimum complexity (as defined in
        :func:`expression_complexity`) of an expression for numexpr to be used.
        This stops numexpr being used for simple things like ``x = y``. The
        default is to use numexpr whenever it is more complicated than this,
        but you can set the threshold higher so that e.g. ``x=y+z`` would
        be excluded as well, which will improve performance when
        operating over small vectors.
        '''
        self.complexity_threshold = complexity_threshold
        
    def translate_expression(self, expr):
        if expression_complexity(expr)>=self.complexity_threshold:
            return '_numexpr.evaluate("'+expr.strip()+'")'
        else:
            return expr.strip()


def expression_complexity(expr):
    '''
    Returns the complexity of the expression defined as the sum of the number
    of leaves and nodes of the tree. TODO: better definition?
    '''
    if isinstance(expr, str):
        expr = sympy.sympify(expr)
    if len(expr.args)==0:
        return 1
    else:
        return 1+sum(map(expression_complexity, expr.args))
    
if __name__=='__main__':
    print expression_complexity('x+y+z')
    print expression_complexity('x+y*(z+1)+2*x**3')
