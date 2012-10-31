from nose.tools import assert_raises

from brian2.equations.equations import Equations
from brian2.stateupdaters.integration import ExplicitStateUpdater, euler, rk2, rk4

def test_explicit_stateupdater_parsing():
    '''
    Test the parsing of explicit state updater descriptions.
    '''
    # These are valid descriptions and should not raise errors
    ExplicitStateUpdater('return x + dt * f(x, t)')
    ExplicitStateUpdater('''x2 = x + dt * f(x, t)
                            return x2''')
    
    # Examples of failed parsing
    # No return statement
    assert_raises(SyntaxError, lambda: ExplicitStateUpdater('x + dt * f(x, t)'))
    # Not an assigment
    assert_raises(SyntaxError, lambda: ExplicitStateUpdater('''2 * x
                                                               return x + dt * f(x, t)'''))

def test_str_repr():
    '''
    Assure that __str__ and __repr__ do not raise errors 
    '''
    for integrator in [euler, rk2, rk4]:
        assert len(str(integrator))
        assert len(repr(integrator))


def test_integrator_code():
    '''
    Check whether the returned abstract code is as expected.
    '''
    # A very simple example where the abstract code should always look the same
    eqs = Equations('dv/dt = -v / (1 * second) : 1')
    
    # Only test very basic stuff (expected number of lines and last line)
    for integrator, lines in zip([euler, rk2, rk4], [2, 3, 6]):
        code_lines = integrator(eqs).split('\n')
        assert len(code_lines) == lines
        assert code_lines[-1] == 'v = _v'


if __name__ == '__main__':
    test_explicit_stateupdater_parsing()
    test_str_repr()
    test_integrator_code()


    
    
    