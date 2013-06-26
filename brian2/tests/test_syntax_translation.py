'''
Tests the brian2.codegen.syntax package
'''
from brian2.utils.stringtools import get_identifiers
from brian2.codegen.ast_parser import (NodeRenderer, NumpyNodeRenderer,
                                       CPPNodeRenderer,
                                       )
from brian2.utils.stringtools import get_identifiers
from numpy.testing import assert_raises, assert_equal
from numpy.random import rand, randint
import numpy as np
try:
    from scipy import weave
except ImportError:
    weave = None
import nose
            

def parse_expressions(renderer, evaluator, numvalues=10):
    # TODO: add some tests with e.g. 1.0%2.0 etc. once this is implemented in C++
    exprs = '''
    a+b+c*d+e-f+g-(b+d)-(a-c)
    a**b**2
    a**(b**2)
    (a**b)**2
    a*(b+c*(a+b)*(a-(c*d)))
    a/b/c-a/(b/c)
    a<b
    a<=b
    a>b
    a>=b
    a==b
    a!=b
    a+1
    1+a
    1+3
    a>0.5 and b>0.5
    a>0.5&b>0.5&c>0.5
    (a>0.5) & (b>0.5) & (c>0.5)
    a>0.5 and b>0.5 or c>0.5
    a>0.5 and b>0.5 or not c>0.5
    2%4
    '''
    exprs = [([m for m in get_identifiers(l) if len(m)==1], [], l.strip()) for l in exprs.split('\n') if l.strip()]
    i, imod = 1, 33
    for varids, funcids, expr in exprs:
        pexpr = renderer.render_expr(expr)
        n = 0
        for _ in xrange(numvalues):
            # assign some random values
            ns = {}
            for v in varids:
                ns[v] = float(i)/imod
                i = i%imod+1
            r1 = eval(expr.replace('&', ' and ').replace('|', ' or '), ns)
            n += 1
            r2 = evaluator(pexpr, ns)
            try:
                assert_equal(r1, r2)
            except AssertionError as e:
                raise AssertionError("In expression "+expr+" translated to "+pexpr+" "+str(e))


def numpy_evaluator(expr, userns):
    ns = {}
    #exec 'from numpy import logical_not' in ns
    ns['logical_not'] = np.logical_not
    ns.update(**userns)
    for k in userns.keys():
        if not k.startswith('_'):
            ns[k] = np.array([userns[k]])
    try:
        x = eval(expr, ns)
    except Exception as e:
        raise ValueError("Could not evaluate numpy expression "+expr+" exception "+str(e))
    if isinstance(x, np.ndarray):
        return x[0]
    else:
        return x
    
    
def cpp_evaluator(expr, ns):
    if weave is not None:
        return weave.inline('return_val = %s;' % expr, ns.keys(), local_dict=ns,
                            compiler='gcc')
    else:
        raise nose.SkipTest('No weave support.')


def test_parse_expressions_python():
    parse_expressions(NodeRenderer(), eval)


def test_parse_expressions_numpy():
    parse_expressions(NumpyNodeRenderer(), numpy_evaluator)


def test_parse_expressions_cpp():
    parse_expressions(CPPNodeRenderer(), cpp_evaluator)


if __name__=='__main__':
    test_parse_expressions_python()
    test_parse_expressions_numpy()
    test_parse_expressions_cpp()
    