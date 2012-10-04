import warnings

import numpy as np
from numpy.testing import assert_raises, assert_equal
import sympy

from brian2 import Expression, Statements
from brian2 import ms, mV, volt, second, get_dimensions, DimensionMismatchError
from brian2.equations.codestrings import ResolutionConflictWarning

import brian2

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
    
    # Only specifying part of the namespace
    expr = Expression('-(v + I) / tau', namespace={'I' : another_I},
                      exhaustive=False)
    # make sure this triggers a warning (the 'I' in the namespace shadows the
    # I variable defined above
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        expr.resolve(['v'])        
        assert len(w) == 1
        assert issubclass(w[0].category, ResolutionConflictWarning)
        
    assert expr.namespace['I'] == another_I and expr.namespace['tau'] == tau
    
    # test resolution of units not present in any namespace
    expr = Expression('v * amp * ohm')
    expr.resolve(['v'])
    assert expr.namespace['ohm'] is brian2.ohm and expr.namespace['amp'] is brian2.amp


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
    assert sympy.sympify(expr_string) == sympy.sympify(str(expr))
    assert sympy.sympify(expr_string) == sympy.sympify(eval(repr(expr)).code)
    
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
    test_frozen()
    test_str_repr()