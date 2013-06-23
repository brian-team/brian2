'''
Tests the brian2.codegen.syntax package
'''
from brian2.utils.stringtools import get_identifiers
from brian2.codegen.ast_parser import (NodeRenderer, NumpyNodeRenderer,
                                       CPPNodeRenderer,
                                       )
from numpy.testing import assert_raises, assert_equal
from numpy.random import rand, randint
import numpy as np
try:
    from scipy import weave
except ImportError:
    weave = None
import nose

def generate_expressions(N=100, numvars=5, numfloats=1, numints=1, complexity=5, depth=3):
    ops = ['+', '*', '-', '/', '**']
    vars = [chr(ord('a')+i) for i in xrange(numvars)]
    consts = [rand() for _ in xrange(numfloats)]+range(1, 1+numints)
    varsconsts = [str(x) for x in vars+consts]
    for _ in xrange(N):
        expr = 'a'
        for _ in xrange(depth):
            s = 'a'
            for _ in xrange(complexity):
                op = ops[randint(len(ops))]
                var = vars[randint(numvars)]
                s = s+op+var
            op = ops[randint(len(ops))]
            expr = '(%s)%s(%s)'%(expr, op, s)
        yield (vars, [], expr)
            

def parse_expressions(renderer, evaluator, numvalues=10):
    exprs = list(generate_expressions())
    additional_exprs = '''
    a<b
    a<=b
    a>b
    a>=b
    a==b
    a!=b
    a+1
    1+a
    a%2
    a%2.0
    1+3
    a>1 and b>1
    '''
    exprs = exprs+[('abc', [], l.strip()) for l in additional_exprs.split('\n') if l.strip()]
    for varids, funcids, expr in exprs:
        pexpr = renderer.render_expr(expr)
        n = 0
        for _ in xrange(numvalues):
            # assign some random values
            ns = dict((v, rand()) for v in varids)
            try:
                r1 = eval(expr, ns)
            except (ZeroDivisionError, ValueError, OverflowError):
                continue
            n += 1
            r2 = evaluator(pexpr, ns)
            assert_equal(r1, r2)


def numpy_evaluator(expr, ns):
    ns = ns.copy()
    for k in ns.keys():
        if not k.startswith('_'):
            ns[k] = np.array([ns[k]])
    x = eval(expr, ns)
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
    # Skipy this test because we haven't handled e.g. 1.2%2 or 2%1.3 yet
    raise nose.SkipTest()
    parse_expressions(CPPNodeRenderer(), cpp_evaluator)


if __name__=='__main__':
    test_parse_expressions_python()
    test_parse_expressions_numpy()
    test_parse_expressions_cpp()
    