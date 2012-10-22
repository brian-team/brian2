import numpy as np
from numpy.testing import assert_raises, assert_equal
import sympy

from brian2 import Expression, Statements
from brian2 import Hz, ms, mV, volt, second, get_dimensions, DimensionMismatchError

from brian2.utils.logger import catch_logs

import brian2

def sympy_equals(expr1, expr2):
    '''
    Test that whether two string expressions are equal using sympy, allowing
    e.g. for ``sympy_equals("x * x", "x ** 2") == True``.
    '''
    s_expr1 = sympy.sympify(expr1).expand()
    s_expr2 = sympy.sympify(expr2).expand()
    return s_expr1 == s_expr2


def test_expr_creation():
    '''
    Test creating expressions.
    '''    
    expr = Expression('v > 5 * mV')
    assert expr.code == 'v > 5 * mV'
    assert not expr.is_resolved
    assert not expr.exhaustive
    assert ('v' in expr.identifiers and 'mV' in expr.identifiers and
            not 'V' in expr.identifiers)
    assert_raises(SyntaxError, lambda: Expression('v 5 * mV'))
    
def test_expr_check_linearity():
    '''
    Test checking for linearity.
    '''
    expr = Expression('-v / tau + sin(2 * pi * t * f)')
    assert expr.check_linearity('v')
    assert expr.check_linearity('x') # does not appear in the expression
    assert not expr.check_linearity('tau')

def test_expr_units():
    '''
    Test getting/checking the units of an expression.
    '''
    tau = 5 * ms
    expr = Expression('-v / tau', namespace={'tau': tau})
    expr.resolve(['v'])
    expr.check_units(volt / second, {'v': volt})
    assert_raises(DimensionMismatchError, lambda: expr.check_units(volt / second,
                                                                   {'v': second}))
    assert_raises(DimensionMismatchError, lambda: expr.check_units(volt,
                                                                   {'v': volt}))
    assert expr.get_dimensions({'v': volt}) == get_dimensions(volt / second)

def test_resolve():
    '''
    Test resolving external identifiers.
    '''
    I = 3 * mV
    tau = 5 * ms
    expr = Expression('-(v + I) / tau')
    expr.resolve(['v'])
    assert expr.is_resolved
    assert not 'v' in expr.namespace
    assert expr.namespace['I'] == I and expr.namespace['tau'] == tau
    
    # trying to resolve a second time should raise an error
    assert_raises(TypeError, lambda: expr.resolve(['v']))
    
    another_I = 5 * mV
    expr = Expression('-(v + I) / tau', namespace={'I' : another_I})
    # tau is not defined, the namespace should be exhaustive
    assert_raises(ValueError, lambda: expr.resolve(['v']))
    expr = Expression('-(v + I) / tau', namespace={'I' : another_I,
                                                   'tau': tau})
    # Now it should work
    expr.resolve(['v'])
    assert expr.namespace['I'] == another_I and expr.namespace['tau'] == tau
    
    # test resolution of units not present in any namespace
    expr = Expression('v * amp * ohm')
    expr.resolve(['v'])
    assert expr.namespace['ohm'] is brian2.ohm and expr.namespace['amp'] is brian2.amp

    # Test that resolving is a prerequisite for a number of functions
    expr = Expression('-v / tau')
    assert_raises(TypeError, expr.frozen)
    assert_raises(TypeError, lambda: expr.eval(['v']))


def test_resolution_warnings():
    '''
    Test that certain calls to resolve generate a warning.
    '''
    I = 3 * mV
    tau = 5 * ms
    another_I = 5 * mV
    # Only specifying part of the namespace
    expr = Expression('-(v + I) / tau', namespace={'I' : another_I},
                      exhaustive=False)
    
    # make sure this triggers a warning (the 'I' in the namespace shadows the
    # I variable defined above
    with catch_logs() as logs:
        expr.resolve(['v'])
        assert len(logs) == 1
        assert logs[0][0] == 'WARNING' 
        assert logs[0][1].endswith('resolution_conflict')
        print 'Message: ', logs[0][2]         
        assert expr.namespace['I'] == another_I and expr.namespace['tau'] == tau
    
    freq = 300 * Hz
    t = 5 * second
    # This expression treats t as a special variable and is not actually using
    # the t above!
    expr = Expression('sin(2 * 3.141 * freq * t)', namespace={'sin': np.sin},
                      exhaustive=False)
    with catch_logs() as logs:
        expr.resolve([])
        assert len(logs) == 1
        assert logs[0][0] == 'WARNING' 
        assert logs[0][1].endswith('resolution_conflict')
        print 'Message: ', logs[0][2]            
        assert expr.namespace['freq'] == freq and not 't' in expr.namespace

    I = 3 * mV
    tau = 5 * ms    
    expr = Expression('-(v + I)/ tau')
    # If we claim that I is an internal variable, it shadows the variable
    # defined in the local namespace -- this should trigger a warning
    with catch_logs() as logs:
        expr.resolve(['v', 'I'])
        assert len(logs) == 1
        assert logs[0][0] == 'WARNING' 
        assert logs[0][1].endswith('resolution_conflict')
        print 'Message: ', logs[0][2]    
        assert expr.namespace['tau'] == tau and not 'I' in expr.namespace
    
    # A more extreme example: I is defined above, but also in the namespace and
    # is claimed to be an internal variable
    expr = Expression('-(v + I)/ tau', namespace={'I': 5 * mV},
                      exhaustive=False)
    with catch_logs() as logs:
        expr.resolve(['v', 'I'])
        assert len(logs) == 1
        assert logs[0][0] == 'WARNING' 
        assert logs[0][1].endswith('resolution_conflict')
        print 'Message: ', logs[0][2]    
        assert expr.namespace['tau'] == tau and not 'I' in expr.namespace


def test_split_stochastic():
    tau = 5 * ms
    expr = Expression('(-v + I) / tau')
    expr.resolve(['v', 'I'])
    # No stochastic part
    assert expr.split_stochastic() == (expr, None)
    
    expr = Expression('(-v + I) / tau + sigma*xi/tau**.5')
    expr.resolve(['v', 'I', 'sigma'])
    non_stochastic, stochastic = expr.split_stochastic()
    assert 'xi' in stochastic.identifiers
    assert sympy_equals(non_stochastic.code, '(-v + I) / tau')
    assert sympy_equals(stochastic.code, 'sigma*xi/tau**.5')
    
    expr = Expression('-v / tau + 1 / xi')
    assert_raises(ValueError, expr.split_stochastic)
    

def test_frozen():
    '''
    Test that freezing a codestring works
    '''
    tau = 5 * ms
    expr = Expression('-v / tau', namespace={'tau': tau})
    expr.resolve(['v'])
    
    frozen_expr = expr.frozen()
    
    assert 'tau' in expr.identifiers and not 'tau' in frozen_expr.identifiers
    
    # When getting rid of units, the frozen expression and the original
    # expression should give the same result
    assert_equal(np.asarray(expr.eval({'v': 1})),
                 np.asarray(frozen_expr.eval({'v': 1})))
    
    # frozen should leave calls to function alone
    def f(x):
        return -x
    expr = Expression('f(v) / tau')
    expr.resolve(['v'])
    frozen_expr = expr.frozen()
    assert 'f' in expr.identifiers and 'f' in frozen_expr.identifiers


def test_str_repr():
    '''
    Test the string representation of expressions and statements. Assumes that
    __str__ returns the complete expression/statement string and __repr__ a
    string of the form "Expression(...)" or "Statements(...)" that can be
    evaluated.
    '''
    expr_string = '(v - I)/ tau'
    expr = Expression(expr_string)
    
    # use sympy to check for equivalence of expressions (terms may have be
    # re-arranged by sympy)
    assert sympy_equals(expr_string, str(expr))
    assert sympy_equals(expr_string, eval(repr(expr)).code)
    
    # Use exact string equivalence for statements
    statement_string = 'v += w'
    statement = Statements(statement_string)
    
    assert str(statement) == 'v += w'
    assert repr(statement) == "Statements('v += w')"

if __name__ == '__main__':
    test_expr_creation()
    test_expr_check_linearity()
    test_expr_units()
    test_resolve()
    test_resolution_warnings()
    test_frozen()
    test_split_stochastic()
    test_str_repr()