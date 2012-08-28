from base import Language
from python import PythonLanguage, PythonCodeObject
import sympy
try:
    import numexpr
except ImportError:
    numexpr = None

__all__ = ['NumexprPythonLanguage', 'NumexprPythonCodeObject']

class NumexprPythonLanguage(PythonLanguage):
    '''
    ``complexity_threshold``
        The minimum complexity (as defined in
        :func:`expression_complexity`) of an expression for numexpr to be used.
        This stops numexpr being used for simple things like ``x = y``. The
        default is to use numexpr whenever it is more complicated than this,
        but you can set the threshold higher so that e.g. ``x=y+z`` would
        be excluded as well, which will improve performance when
        operating over small vectors.
    
    ``multicore``
        Whether or not to use multiple cores in numexpr evaluation, set
        to True, False or number of cores, including negative numbers
        for number_of_cores()-1, etc.
    '''
    def __init__(self, complexity_threshold=2, multicore=True):
        self.complexity_threshold = complexity_threshold
        nc = numexpr.detect_number_of_cores()
        if multicore is True:
            multicore = nc
        elif multicore is False:
            multicore = 1
        elif multicore<=0:
            multicore += nc
        numexpr.set_num_threads(multicore)
        
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
        
    def translate_statement(self, statement):
        if expression_complexity(statement.expr)<self.complexity_threshold:
            return PythonLanguage.translate_statement(self, statement)
        if statement.op==':=':
            return PythonLanguage.translate_statement(self, statement)
        if statement.op=='=':
            # statement.op=='=', we use inplace form of numexpr
            statement.inplace = True
            return '_numexpr.evaluate("{expr}", out={var})'.format(
                                        expr=statement.expr, var=statement.var)
        # TODO: why does this not work in example_state_update.py??
        # I don't know, but it works fine if N=1 and not if N>1 therefore I
        # suspect a bug with numexpr?
        return PythonLanguage.translate_statement(self, statement)
#        # other statement.op is [?]=, e.g. +=, *=, **=, /=
#        opfirst = statement.op[:-1]
#        return '_numexpr.evaluate("{var}{opfirst}({expr})", out={var})'.format(
#                        var=statement.var, opfirst=opfirst, expr=statement.expr)

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
        PythonCodeObject.compile(self, namespace)
        exec 'import numexpr as _numexpr' in self.namespace
    
    
if __name__=='__main__':
    print expression_complexity('x+y+z')
    print expression_complexity('x+y*(z+1)+2*x**3')
