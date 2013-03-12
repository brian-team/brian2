from nose.tools import assert_raises
from numpy.testing.utils import assert_equal

from brian2 import *

def test_explicit_stateupdater_parsing():
    '''
    Test the parsing of explicit state updater descriptions.
    '''
    # These are valid descriptions and should not raise errors
    ExplicitStateUpdater('return x + dt * f(x, t)', priority=10)
    ExplicitStateUpdater('''x2 = x + dt * f(x, t)
                            return x2''', priority=10)
    
    # Examples of failed parsing
    # No return statement
    assert_raises(SyntaxError, lambda: ExplicitStateUpdater('x + dt * f(x, t)', 10))
    # Not an assigment
    assert_raises(SyntaxError, lambda: ExplicitStateUpdater('''2 * x
                                                               return x + dt * f(x, t)''', 10))

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

def test_priority():
    updater = ExplicitStateUpdater('return x + dt * f(x, t)', priority=10)
    # Equations that work for the state updater
    eqs = Equations('dv/dt = -v / (10*ms) : 1')
    # namespace and specifiers should not be necessary here
    namespace = {}
    specifiers = {} 
    assert updater.get_priority(eqs, namespace, specifiers) == 10
    
    eqs = Equations('dv/dt = -v / (10*ms) + xi/(10*ms)**.5 : 1')
    assert updater.get_priority(eqs, namespace, specifiers) == 0

def test_static_equations():
    '''
    Make sure that the integration of a (non-stochastic) differential equation
    does not depend on whether it's formulated using static equations.
    '''
    # no static equation
    eqs1 = 'dv/dt = (-v + sin(2*pi*100*Hz*t)) / (10*ms) : 1'
    # same with static equation
    eqs2 = '''dv/dt = I / (10*ms) : 1
              I = -v + sin(2*pi*100*Hz*t): 1'''
    
    methods = ['euler', 'rk2', 'rk4']
    for method in methods:
        G1 = NeuronGroup(1, eqs1, clock=Clock(), method=method)
        G1.v = 1
        G2 = NeuronGroup(1, eqs2, clock=Clock(), method=method)
        G2.v = 1
        mon1 = StateMonitor(G1, 'v', record=True)
        mon2 = StateMonitor(G2, 'v', record=True)
        net1 = Network(G1, mon1)
        net2 = Network(G2, mon2)
        net1.run(10*ms)
        net2.run(10*ms)
        assert_equal(mon1.v, mon2.v, 'Results for method %s differed!' % method)


if __name__ == '__main__':
    test_explicit_stateupdater_parsing()
    test_str_repr()
    test_integrator_code()
    test_priority()
    test_static_equations()
