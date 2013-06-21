'''
Tests the brian2.codegen.syntax package
'''
from brian2.utils.stringtools import get_identifiers
from brian2.codegen.syntax.parser import parse_expr, parse_statement
from numpy.testing import assert_raises, assert_equal
from numpy.random import rand, randint

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
            
def test_parse_expressions(numvalues=10):
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
        pexpr = str(parse_expr(expr))
        n = 0
        for _ in xrange(numvalues):
            # assign some random values
            ns = dict((v, rand()) for v in varids)
            try:
                r1 = eval(expr, ns)
            except (ZeroDivisionError, ValueError, OverflowError):
                continue
            n += 1
            r2 = eval(pexpr, ns)
            assert_equal(r1, r2)
#        print n

if __name__=='__main__':
    test_parse_expressions()
    