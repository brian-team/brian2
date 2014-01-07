import re
from collections import namedtuple

from numpy.testing.utils import assert_equal, assert_raises

from brian2 import *
from brian2.utils.logger import catch_logs
from brian2.core.variables import ArrayVariable, AttributeVariable, Variable


def test_explicit_stateupdater_parsing():
    '''
    Test the parsing of explicit state updater descriptions.
    '''
    # These are valid descriptions and should not raise errors
    updater = ExplicitStateUpdater('x_new = x + dt * f(x, t)')
    updater(Equations('dv/dt = -v / tau : 1'))
    updater = ExplicitStateUpdater('''x2 = x + dt * f(x, t)
                                      x_new = x2''')
    updater(Equations('dv/dt = -v / tau : 1'))
    updater = ExplicitStateUpdater('''x1 = g(x, t) * dW
                                      x2 = x + dt * f(x, t)
                                      x_new = x1 + x2''')
    updater(Equations('dv/dt = -v / tau + v * xi * tau**-.5: 1'))
    
    updater = ExplicitStateUpdater('''x_support = x + dt*f(x, t) + dt**.5 * g(x, t)
                                      g_support = g(x_support, t)
                                      k = 1/(2*dt**.5)*(g_support - g(x, t))*(dW**2)
                                      x_new = x + dt*f(x,t) + g(x, t) * dW + k''')
    updater(Equations('dv/dt = -v / tau + v * xi * tau**-.5: 1'))

    
    # Examples of failed parsing
    # No x_new = ... statement
    assert_raises(SyntaxError, lambda: ExplicitStateUpdater('x = x + dt * f(x, t)'))
    # Not an assigment
    assert_raises(SyntaxError, lambda: ExplicitStateUpdater('''2 * x
                                                               x_new = x + dt * f(x, t)'''))
    
    # doesn't separate into stochastic and non-stochastic part
    updater = ExplicitStateUpdater('''x_new = x + dt * f(x, t) * g(x, t) * dW''')
    assert_raises(ValueError, lambda: updater(Equations('')))

def test_str_repr():
    '''
    Assure that __str__ and __repr__ do not raise errors 
    '''
    for integrator in [linear, euler, rk2, rk4]:
        assert len(str(integrator))
        assert len(repr(integrator))


def test_integrator_code():
    '''
    Check whether the returned abstract code is as expected.
    '''
    # A very simple example where the abstract code should always look the same
    eqs = Equations('dv/dt = -v / (1 * second) : 1')
    
    # Only test very basic stuff (expected number of lines and last line)
    for integrator, lines in zip([linear, euler, rk2, rk4], [2, 2, 3, 6]):
        code_lines = integrator(eqs).split('\n')
        err_msg = 'Returned code for integrator %s had %d lines instead of %d' % (integrator.__class__.__name__, len(code_lines), lines)
        assert len(code_lines) == lines, err_msg
        assert code_lines[-1] == 'v = _v'
    
    # Make sure that it isn't a problem to use 'x', 'f' and 'g'  as variable
    # names, even though they are also used in state updater descriptions.
    # The resulting code should be identical when replacing x by v (and ..._x by
    # ..._v)
    for varname in ['x', 'f', 'g']:
        eqs_v = Equations('dv/dt = -v / (1 * second) : 1')
        eqs_var = Equations('d{varname}/dt = -{varname} / (1 * second) : 1'.format(varname=varname))  
        for integrator in [linear, euler, rk2, rk4]:
            code_v = integrator(eqs_v)
            code_var = integrator(eqs_var)
            # Re-substitute the variable names in the output
            code_var = re.sub(r'\b{varname}\b'.format(varname=varname),
                              'v', code_var)
            code_var = re.sub(r'\b(\w*)_{varname}\b'.format(varname=varname),
                              r'\1_v', code_var)
            assert code_var == code_v


def test_priority():
    updater = ExplicitStateUpdater('x_new = x + dt * f(x, t)')
    # Equations that work for the state updater
    eqs = Equations('dv/dt = -v / (10*ms) : 1')
    # Fake clock class
    MyClock = namedtuple('MyClock', ['t_', 'dt_'])
    clock = MyClock(t_=0, dt_=0.0001)
    variables = {'v': ArrayVariable('v', Unit(1), None, constant=False),
                  't': AttributeVariable(second, clock, 't_', constant=False),
                  'dt': AttributeVariable(second, clock, 'dt_', constant=True)}
    assert updater.can_integrate(eqs, variables)

    # Non-constant parameter in the coefficient, linear integration does not
    # work
    eqs = Equations('''dv/dt = -param * v / (10*ms) : 1
                       param : 1''')
    variables['param'] = ArrayVariable('param', Unit(1), None, constant=False)
    assert updater.can_integrate(eqs, variables)
    can_integrate = {linear: False, euler: True, rk2: True, rk4: True, 
                     milstein: True}

    for integrator, able in can_integrate.iteritems():
        assert integrator.can_integrate(eqs, variables) == able

    # Constant parameter in the coefficient, linear integration should
    # work
    eqs = Equations('''dv/dt = -param * v / (10*ms) : 1
                       param : 1 (constant)''')
    variables['param'] = ArrayVariable('param', Unit(1), None, constant=True)
    assert updater.can_integrate(eqs, variables)
    can_integrate = {linear: True, euler: True, rk2: True, rk4: True, 
                     milstein: True}
    del variables['param']

    for integrator, able in can_integrate.iteritems():
        assert integrator.can_integrate(eqs, variables) == able

    # External parameter in the coefficient, linear integration *should* work
    # (external parameters don't change during a run)
    param = 1
    eqs = Equations('dv/dt = -param * v / (10*ms) : 1')
    assert updater.can_integrate(eqs, variables)
    can_integrate = {linear: True, euler: True, rk2: True, rk4: True, 
                     milstein: True}
    for integrator, able in can_integrate.iteritems():
        assert integrator.can_integrate(eqs, variables) == able
    
    # Equation with additive noise
    eqs = Equations('dv/dt = -v / (10*ms) + xi/(10*ms)**.5 : 1')
    assert not updater.can_integrate(eqs, variables)
    
    can_integrate = {linear: False, euler: True, rk2: False, rk4: False, 
                     milstein: True}
    for integrator, able in can_integrate.iteritems():
        assert integrator.can_integrate(eqs, variables) == able
    
    # Equation with multiplicative noise
    eqs = Equations('dv/dt = -v / (10*ms) + v*xi/(10*ms)**.5 : 1')
    assert not updater.can_integrate(eqs, variables)
    
    can_integrate = {linear: False, euler: False, rk2: False, rk4: False, 
                     milstein: True}
    for integrator, able in can_integrate.iteritems():
        assert integrator.can_integrate(eqs, variables) == able
    

def test_registration():
    '''
    Test state updater registration.
    '''
    # Save state before tests
    before = list(StateUpdateMethod.stateupdaters)
    
    lazy_updater = ExplicitStateUpdater('x_new = x')
    StateUpdateMethod.register('lazy', lazy_updater)
    
    # Trying to register again
    assert_raises(ValueError,
                  lambda: StateUpdateMethod.register('lazy', lazy_updater))
    
    # Trying to register something that is not a state updater
    assert_raises(ValueError,
                  lambda: StateUpdateMethod.register('foo', 'just a string'))
    
    # Trying to register with an invalid index
    assert_raises(TypeError,
                  lambda: StateUpdateMethod.register('foo', lazy_updater,
                                                     index='not an index'))
    
    # reset to state before the test
    StateUpdateMethod.stateupdaters = before 


def test_determination():
    '''
    Test the determination of suitable state updaters.
    '''
    
    # To save some typing
    determine_stateupdater = StateUpdateMethod.determine_stateupdater
    
    # Save state before tests
    before = list(StateUpdateMethod.stateupdaters)
    
    eqs = Equations('dv/dt = -v / (10*ms) : 1')
    # Just make sure that state updaters know about the two state variables
    variables = {'v': Variable(unit=None), 'w': Variable(unit=None)}
    
    # all methods should work for these equations.
    # First, specify them explicitly (using the object)
    for integrator in (linear, euler, exponential_euler, #TODO: Removed "independent" here due to the issue in sympy 0.7.4
                       rk2, rk4, milstein):
        with catch_logs() as logs:
            returned = determine_stateupdater(eqs, variables,
                                              method=integrator)
            assert returned is integrator, 'Expected state updater %s, got %s' % (integrator, returned)
            assert len(logs) == 0, 'Got %d unexpected warnings: %s' % (len(logs), str([l[2] for l in logs]))
    
    # Equation with multiplicative noise, only milstein should work without
    # a warning
    eqs = Equations('dv/dt = -v / (10*ms) + v*xi*second**-.5: 1')
    for integrator in (linear, independent, euler, exponential_euler, rk2, rk4):
        with catch_logs() as logs:
            returned = determine_stateupdater(eqs, variables,
                                              method=integrator)
            assert returned is integrator, 'Expected state updater %s, got %s' % (integrator, returned)
            # We should get a warning here
            assert len(logs) == 1, 'Got %d warnings but expected 1: %s' % (len(logs), str([l[2] for l in logs]))
            
    with catch_logs() as logs:
        returned = determine_stateupdater(eqs, variables,
                                          method=milstein)
        assert returned is milstein, 'Expected state updater milstein, got %s' % (integrator, returned)
        # No warning here
        assert len(logs) == 0, 'Got %d unexpected warnings: %s' % (len(logs), str([l[2] for l in logs]))
    
    
    # Arbitrary functions (converting equations into abstract code) should
    # always work
    my_stateupdater = lambda eqs: 'x_new = x'
    with catch_logs() as logs:
        returned = determine_stateupdater(eqs, variables,
                                          method=my_stateupdater)
        assert returned is my_stateupdater
        # No warning here
        assert len(logs) == 0
    
    
    # Specification with names
    eqs = Equations('dv/dt = -v / (10*ms) : 1')
    for name, integrator in [('linear', linear), ('euler', euler),
                             #('independent', independent), #TODO: Removed "independent" here due to the issue in sympy 0.7.4
                             ('exponential_euler', exponential_euler),
                             ('rk2', rk2), ('rk4', rk4),
                             ('milstein', milstein)]:
        with catch_logs() as logs:
            returned = determine_stateupdater(eqs, variables,
                                              method=name)
        assert returned is integrator
        # No warning here
        assert len(logs) == 0    

    # Now all except milstein should refuse to work
    eqs = Equations('dv/dt = -v / (10*ms) + v*xi*second**-.5: 1')
    for name in ['linear', 'independent', 'euler', 'exponential_euler',
                 'rk2', 'rk4']:
        assert_raises(ValueError, lambda: determine_stateupdater(eqs,
                                                                 variables,
                                                                 method=name))
    # milstein should work
    with catch_logs() as logs:
        determine_stateupdater(eqs, variables, method='milstein')
        assert len(logs) == 0
    
    # non-existing name
    assert_raises(ValueError, lambda: determine_stateupdater(eqs,
                                                             variables,
                                                             method='does_not_exist'))
    
    # Automatic state updater choice should return linear for linear equations,
    # euler for non-linear, non-stochastic equations and equations with
    # additive noise, milstein for equations with multiplicative noise
    eqs = Equations('dv/dt = -v / (10*ms) : 1')
    assert determine_stateupdater(eqs, variables) is linear
    
    # This is conditionally linear
    eqs = Equations('''dv/dt = -(v + w**2)/ (10*ms) : 1
                       dw/dt = -w/ (10*ms) : 1''')
    assert determine_stateupdater(eqs, variables) is exponential_euler

    eqs = Equations('dv/dt = sin(t) / (10*ms) : 1')
    assert determine_stateupdater(eqs, variables) is independent

    eqs = Equations('dv/dt = -sqrt(v) / (10*ms) : 1')
    assert determine_stateupdater(eqs, variables) is euler

    eqs = Equations('dv/dt = -v / (10*ms) + 0.1*second**-.5*xi: 1')
    assert determine_stateupdater(eqs, variables) is euler

    eqs = Equations('dv/dt = -v / (10*ms) + v*0.1*second**-.5*xi: 1')
    assert determine_stateupdater(eqs, variables) is milstein

    # remove all registered state updaters --> automatic choice no longer works
    StateUpdateMethod.stateupdaters = {}
    assert_raises(ValueError, lambda: determine_stateupdater(eqs, variables))

    # reset to state before the test
    StateUpdateMethod.stateupdaters = before

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
    
    methods = ['euler', 'exponential_euler', 'rk2', 'rk4']
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
    test_determination()
    test_explicit_stateupdater_parsing()
    test_str_repr()
    test_integrator_code()
    test_priority()
    test_registration()
    test_static_equations()
