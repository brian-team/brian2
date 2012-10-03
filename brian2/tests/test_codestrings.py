import warnings

from numpy.testing import assert_raises

from brian2 import Expression
from brian2 import ms, mV, volt, second, get_dimensions, DimensionMismatchError
from brian2.equations.codestrings import ResolutionConflictWarning

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
    
if __name__ == '__main__':
    test_expr_creation()
    test_expr_check_linearity()
    test_expr_units()
    test_resolve()