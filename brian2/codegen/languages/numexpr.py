from base import Language
from python import PythonLanguage, PythonCodeObject
import sympy

__all__ = ['NumexprPythonLanguage', 'NumexprPythonCodeObject']

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
        
    # TODO: there is now an out argument in numexpr, so we can do:
    #   numexpr.evaluate('y+z', {'y':y, 'z':z}, out=x)
    # for maximum efficiency. Once this is done, we also need to consider
    # whether we do all operations in-place or rebind, and handle correctly
    # the other parts of code generation.
    def translate_expression(self, expr):
        if expression_complexity(expr)>=self.complexity_threshold:
            return '_numexpr.evaluate("'+expr.strip()+'")'
        else:
            return expr.strip()

    def code_object(self, code):
        return NumexprPythonCodeObject(code)


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


class NumexprPythonCodeObject(PythonCodeObject):
    def compile(self, namespace):
        #self.code = 'import numexpr as _numexpr\n'+self.code
        PythonCodeObject.compile(self, namespace)
        exec 'import numexpr as _numexpr' in self.namespace
#        import numexpr as _numexpr
#        self.namespace['_numexpr'] = _numexpr
    
    
if __name__=='__main__':
    print expression_complexity('x+y+z')
    print expression_complexity('x+y*(z+1)+2*x**3')
