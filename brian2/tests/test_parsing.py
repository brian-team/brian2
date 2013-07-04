'''
Tests the brian2.parsing package
'''
from brian2.utils.stringtools import get_identifiers
from brian2.parsing.rendering import (NodeRenderer, NumpyNodeRenderer,
                                      CPPNodeRenderer,
                                      )
from brian2.parsing.dependencies import abstract_code_dependencies

from numpy.testing import assert_allclose

import numpy as np
from brian2.codegen.parsing import str_to_sympy, sympy_to_str

try:
    from scipy import weave
except ImportError:
    weave = None
import nose

# TODO: add some tests with e.g. 1.0%2.0 etc. once this is implemented in C++
TEST_EXPRESSIONS = '''
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
    a>0.5 and b>0.5 or c>0.5
    a>0.5 and b>0.5 or not c>0.5
    2%4
    '''


def parse_expressions(renderer, evaluator, numvalues=10):
    exprs = [([m for m in get_identifiers(l) if len(m)==1], [], l.strip())
             for l in TEST_EXPRESSIONS.split('\n') if l.strip()]
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
                # Use all close because we can introduce small numerical
                # difference through sympy's rearrangements
                assert_allclose(r1, r2)
            except AssertionError as e:
                raise AssertionError("In expression " + str(expr) +
                                     " translated to " + str(pexpr) +
                                     " " + str(e))


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


def test_parse_expressions_sympy():
    # sympy is about symbolic calculation, the string returned by the renderer
    # contains "Symbol('a')" etc. so we cannot simply evaluate it in a
    # namespace.
    # We therefore use a different approach: Convert the expression to a
    # sympy expression via str_to_sympy (uses the SympyNodeRenderer internally),
    # then convert it back to a string via sympy_to_str and evaluate it

    class SympyRenderer(object):
        def render_expr(self, expr):
            return str_to_sympy(expr)

    def evaluator(expr, ns):
        expr = sympy_to_str(expr)
        return eval(expr, ns)

    parse_expressions(SympyRenderer(), evaluator)


def test_abstract_code_dependencies():
    code = '''
    a = b+c
    d = b+c
    a = func_a()
    a = func_b()
    a = e+d
    '''
    known_vars = set(['a', 'b', 'c'])
    known_funcs = set(['func_a'])
    res = abstract_code_dependencies(code, known_vars, known_funcs)
    expected_res = dict(
        all=['a', 'b', 'c', 'd', 'e',
             'func_a', 'func_b',
             ],
        read=['b', 'c', 'd', 'e'],
        write=['a', 'd'],
        funcs=['func_a', 'func_b'],
        unknown_read=['d', 'e'],
        unknown_write=['d'],
        unknown_funcs=['func_b'],
        undefined_read=['e'],
        newly_defined=['d'],
        )
    for k, v in expected_res.items():
        if not getattr(res, k)==set(v):
            raise AssertionError("For '%s' result is %s expected %s" % (
                                        k, getattr(res, k), set(v)))

if __name__=='__main__':
    test_parse_expressions_python()
    test_parse_expressions_numpy()
    test_parse_expressions_cpp()
    test_parse_expressions_sympy()
    test_abstract_code_dependencies()

