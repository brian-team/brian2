import re
import logging

from numpy.testing.utils import assert_equal, assert_raises, assert_allclose
from nose.plugins.attrib import attr
from nose import with_setup

from brian2 import *
from brian2.utils.logger import catch_logs
from brian2.core.variables import ArrayVariable, Variable, Constant
from brian2.devices.device import reinit_devices
from brian2.stateupdaters.base import UnsupportedEquationsException

@attr('codegen-independent')
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
                                      x_new = x1 + x2''',
                                   stochastic='multiplicative')
    updater(Equations('dv/dt = -v / tau + v * xi * tau**-.5: 1'))
    
    updater = ExplicitStateUpdater('''x_support = x + dt*f(x, t) + dt**.5 * g(x, t)
                                      g_support = g(x_support, t)
                                      k = 1/(2*dt**.5)*(g_support - g(x, t))*(dW**2)
                                      x_new = x + dt*f(x,t) + g(x, t) * dW + k''',
                                   stochastic='multiplicative')
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

@attr('codegen-independent')
def test_non_autonomous_equations():
    # Check that non-autonmous equations are handled correctly in multi-step
    # updates
    updater = ExplicitStateUpdater('x_new = f(x, t + 0.5*dt)')
    update_step = updater(Equations('dv/dt = t : 1'))  # Not a valid equation but...
    # very crude test
    assert '0.5*dt' in update_step

@attr('codegen-independent')
def test_str_repr():
    '''
    Assure that __str__ and __repr__ do not raise errors 
    '''
    for integrator in [linear, euler, rk2, rk4]:
        assert len(str(integrator))
        assert len(repr(integrator))


@attr('codegen-independent')
def test_multiple_noise_variables_basic():
    # Very basic test, only to make sure that stochastic state updaters handle
    # multiple noise variables at all
    eqs = Equations('''dv/dt = -v / (10*ms) + xi_1 * ms ** -.5 : 1
                       dw/dt = -w / (10*ms) + xi_2 * ms ** -.5 : 1''')
    for method in [euler, heun, milstein]:
        code = method(eqs, {})
        assert 'xi_1' in code
        assert 'xi_2' in code


def test_multiple_noise_variables_extended():
    # Some actual simulations with multiple noise variables
    eqs = '''dx/dt = y : 1
             dy/dt = - 1*ms**-1*y - 40*ms**-2*x : Hz
            '''
    all_eqs_noise = ['''dx/dt = y : 1
                        dy/dt = noise_factor*ms**-1.5*xi_1 + noise_factor*ms**-1.5*xi_2
                           - 1*ms**-1*y - 40*ms**-2*x : Hz
                     ''',
                     '''dx/dt = y + noise_factor*ms**-0.5*xi_1: 1
                        dy/dt = noise_factor*ms**-1.5*xi_2
                            - 1*ms**-1*y - 40*ms**-2*x : Hz
                     ''']
    G = NeuronGroup(2, eqs, method='euler')
    G.x = [0.5, 1]
    G.y = [0, 0.5] * Hz
    mon = StateMonitor(G, ['x', 'y'], record=True)
    net = Network(G, mon)
    net.run(10*ms)
    no_noise_x, no_noise_y = mon.x[:], mon.y[:]

    for eqs_noise in all_eqs_noise:
        for method_name, method in [('euler', euler), ('heun', heun)]:
            with catch_logs('WARNING'):
                G = NeuronGroup(2, eqs_noise, method=method)
                G.x = [0.5, 1]
                G.y = [0, 0.5] * Hz
                mon = StateMonitor(G, ['x', 'y'], record=True)
                net = Network(G, mon)
                # We run it deterministically, but still we'd detect major errors (e.g.
                # non-stochastic terms that are added twice, see #330
                net.run(10*ms, namespace={'noise_factor': 0})
            assert_allclose(mon.x[:], no_noise_x,
                            err_msg='Method %s gave incorrect results' % method_name)
            assert_allclose(mon.y[:], no_noise_y,
                            err_msg='Method %s gave incorrect results' % method_name)


old_randn = None
def store_randn():
    global old_randn
    old_randn = DEFAULT_FUNCTIONS['randn']
def restore_randn():
    DEFAULT_FUNCTIONS['randn'] = old_randn

# The "random" values are always 0.5
@implementation('cpp',
                '''
                double randn(int vectorisation_idx)
                {
                    return 0.5;
                }
                ''')
@implementation('cython',
                '''
                cdef double randn(int vectorisation_idx):
                    return 0.5
                ''')
@check_units(N=Unit(1), result=Unit(1))
def fake_randn(N):
    return 0.5*ones(N)

@with_setup(setup=store_randn, teardown=restore_randn)
def test_multiple_noise_variables_deterministic_noise():

    DEFAULT_FUNCTIONS['randn'] = fake_randn

    all_eqs = ['''dx/dt = y : 1
                          dy/dt = -y / (10*ms) + dt**-.5*0.5*ms**-1.5 + dt**-.5*0.5*ms**-1.5: Hz
                     ''',
                     '''dx/dt = y + dt**-.5*0.5*ms**-0.5: 1
                        dy/dt = -y / (10*ms) + dt**-.5*0.5 * ms**-1.5 : Hz
                ''']
    all_eqs_noise = ['''dx/dt = y : 1
                          dy/dt = -y / (10*ms) + xi_1 * ms**-1.5 + xi_2 * ms**-1.5: Hz
                     ''',
                     '''dx/dt = y + xi_1*ms**-0.5: 1
                        dy/dt = -y / (10*ms) + xi_2 * ms**-1.5 : Hz
                     ''']
    for eqs, eqs_noise in zip(all_eqs, all_eqs_noise):
        G = NeuronGroup(2, eqs, method='euler')
        G.x = [5,  17]
        G.y = [25, 5 ] * Hz
        mon = StateMonitor(G, ['x', 'y'], record=True)
        net = Network(G, mon)
        net.run(10*ms)
        no_noise_x, no_noise_y = mon.x[:], mon.y[:]

        for method_name, method in [('euler', euler), ('heun', heun)]:
            with catch_logs('WARNING'):
                G = NeuronGroup(2, eqs_noise, method=method)
                G.x = [5,  17]
                G.y = [25, 5 ] * Hz
                mon = StateMonitor(G, ['x', 'y'], record=True)
                net = Network(G, mon)
                net.run(10*ms)
            assert_allclose(mon.x[:], no_noise_x,
                            err_msg='Method %s gave incorrect results' % method_name)
            assert_allclose(mon.y[:], no_noise_y,
                            err_msg='Method %s gave incorrect results' % method_name)


@with_setup(setup=store_randn, teardown=restore_randn)
def test_pure_noise_deterministic():
    DEFAULT_FUNCTIONS['randn'] = fake_randn
    sigma = 3
    eqs = Equations('dx/dt = sigma*xi/sqrt(ms) : 1')
    dt = 0.1*ms
    for method in ['euler', 'heun', 'milstein']:
        G = NeuronGroup(1, eqs, dt=dt, method=method)
        run(10*dt)
        assert_allclose(G.x, sqrt(dt)*sigma*0.5/sqrt(1*ms)*10,
                        err_msg='method %s did not give the expected result' % method)


@attr('codegen-independent')
def test_temporary_variables():
    '''
    Make sure that the code does the distinction between temporary variables
    in the state updater description and external variables used in the
    equations.
    '''
    # Use a variable name that is used in the state updater description
    k_2 = 5
    eqs = Equations('dv/dt = -(v + k_2)/(10*ms) : 1')
    converted = rk4(eqs)

    # Use a non-problematic name
    k_var = 5
    eqs = Equations('dv/dt = -(v + k_var)/(10*ms) : 1')
    converted2 = rk4(eqs)

    # Make sure that the two formulations result in the same code
    assert converted == converted2.replace('k_var', 'k_2')


@attr('codegen-independent')
def test_temporary_variables2():
    '''
    Make sure that the code does the distinction between temporary variables
    in the state updater description and external variables used in the
    equations.
    '''
    tau = 10*ms
    # Use a variable name that is used in the state updater description
    k = 5
    eqs = Equations('dv/dt = -v/tau + k*xi*tau**-0.5: 1')
    converted = milstein(eqs)

    # Use a non-problematic name
    k_var = 5
    eqs = Equations('dv/dt = -v/tau + k_var*xi*tau**-0.5: 1')
    converted2 = milstein(eqs)

    # Make sure that the two formulations result in the same code
    assert converted == converted2.replace('k_var', 'k')


@attr('codegen-independent')
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
    # The resulting code should be identical when replacing x by x0 (and ..._x by
    # ..._x0)
    for varname in ['x', 'f', 'g']:
        # We use a very similar names here to avoid slightly re-arranged
        # expressions due to alphabetical sorting of terms in
        # multiplications, etc.
        eqs_v = Equations('d{varname}0/dt = -{varname}0 / (1 * second) : 1'.format(varname=varname))
        eqs_var = Equations('d{varname}/dt = -{varname} / (1 * second) : 1'.format(varname=varname))  
        for integrator in [linear, euler, rk2, rk4]:
            code_v = integrator(eqs_v)
            code_var = integrator(eqs_var)
            # Re-substitute the variable names in the output
            code_var = re.sub(r'\b{varname}\b'.format(varname=varname),
                              '{varname}0'.format(varname=varname), code_var)
            code_var = re.sub(r'\b(\w*)_{varname}\b'.format(varname=varname),
                              r'\1_{varname}0'.format(varname=varname), code_var)
            assert code_var == code_v, "'%s' does not match '%s'" % (code_var, code_v)


@attr('codegen-independent')
def test_integrator_code2():
    '''
    Test integration for a simple model with several state variables.
    '''
    eqs = Equations('''
    dv/dt=(ge+gi-v)/tau : volt
    dge/dt=-ge/taue : volt
    dgi/dt=-gi/taui : volt
    ''')
    euler_integration = euler(eqs)
    lines = sorted(euler_integration.split('\n'))
    # Do a very basic check that the right variables are used in every line
    for varname, line in zip(['_ge', '_gi', '_v', 'ge', 'gi', 'v'], lines):
        assert line.startswith(varname + ' = '), 'line "%s" does not start with %s' % (line, varname)
    for variables, line in zip([['dt', 'ge', 'taue'],
                                ['dt', 'gi', 'taui'],
                                ['dt', 'ge', 'gi', 'v', 'tau'],
                                ['_ge'], ['_gi'], ['_v']],
                               lines):
        rhs = line.split('=')[1]
        for variable in variables:
            assert variable in rhs, '%s not in RHS: "%s"' % (variable, rhs)

@attr('codegen-independent')
def test_illegal_calls():
    eqs = Equations('dv/dt = -v / (10*ms) : 1')
    clock = Clock(dt=0.1*ms)
    variables = {'v': ArrayVariable(name='name', unit=Unit(1), size=10,
                                    owner=None, device=None, dtype=np.float64,
                                    constant=False),
                 't': clock.variables['t'],
                 'dt': clock.variables['dt']}
    assert_raises(TypeError, lambda: StateUpdateMethod.apply_stateupdater(eqs,
                                                                          variables,
                                                                          object()))
    assert_raises(TypeError, lambda: StateUpdateMethod.apply_stateupdater(eqs,
                                                                          variables,
                                                                          group_name='my_name',
                                                                          method=object()))
    assert_raises(TypeError, lambda: StateUpdateMethod.apply_stateupdater(eqs,
                                                                          variables,
                                                                          [object(), 'euler']))
    assert_raises(TypeError, lambda: StateUpdateMethod.apply_stateupdater(eqs,
                                                                          variables,
                                                                          group_name='my_name',
                                                                          method=[object(), 'euler']))

@attr('codegen-independent')
def test_priority():
    updater = ExplicitStateUpdater('x_new = x + dt * f(x, t)')
    # Equations that work for the state updater
    eqs = Equations('dv/dt = -v / (10*ms) : 1')
    clock = Clock(dt=0.1*ms)
    variables = {'v': ArrayVariable(name='name', unit=Unit(1), size=10,
                                    owner=None, device=None, dtype=np.float64,
                                    constant=False),
                 't': clock.variables['t'],
                 'dt': clock.variables['dt']}
    updater(eqs, variables)  # should not raise an error

    # External parameter in the coefficient, linear integration should work
    param = 1
    eqs = Equations('dv/dt = -param * v / (10*ms) : 1')
    updater(eqs, variables)  # should not raise an error
    can_integrate = {linear: True, euler: True, rk2: True, rk4: True, 
                     heun: True, milstein: True}
    for integrator, able in can_integrate.iteritems():
        try:
            integrator(eqs, variables)
            if not able:
                raise AssertionError('Should not be able to integrate these '
                                     'equations')
        except UnsupportedEquationsException:
            if able:
                raise AssertionError('Should be able to integrate these '
                                     'equations')
    
    # Equation with additive noise
    eqs = Equations('dv/dt = -v / (10*ms) + xi/(10*ms)**.5 : 1')
    assert_raises(UnsupportedEquationsException,
                  lambda: updater(eqs, variables))
    
    can_integrate = {linear: False, euler: True, rk2: False, rk4: False, 
                     heun: True, milstein: True}
    for integrator, able in can_integrate.iteritems():
        try:
            integrator(eqs, variables)
            if not able:
                raise AssertionError('Should not be able to integrate these '
                                     'equations')
        except UnsupportedEquationsException:
            if able:
                raise AssertionError('Should be able to integrate these '
                                     'equations')
    
    # Equation with multiplicative noise
    eqs = Equations('dv/dt = -v / (10*ms) + v*xi/(10*ms)**.5 : 1')
    assert_raises(UnsupportedEquationsException,
                  lambda: updater(eqs, variables))
    
    can_integrate = {linear: False, euler: False, rk2: False, rk4: False, 
                     heun: True, milstein: True}
    for integrator, able in can_integrate.iteritems():
        try:
            integrator(eqs, variables)
            if not able:
                raise AssertionError('Should not be able to integrate these '
                                     'equations')
        except UnsupportedEquationsException:
            if able:
                raise AssertionError('Should be able to integrate these '
                                     'equations')
    

@attr('codegen-independent')
def test_registration():
    '''
    Test state updater registration.
    '''
    # Save state before tests
    before = dict(StateUpdateMethod.stateupdaters)
    
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


@attr('codegen-independent')
def test_determination():
    '''
    Test the determination of suitable state updaters.
    '''
    # To save some typing
    apply_stateupdater = StateUpdateMethod.apply_stateupdater
    
    eqs = Equations('dv/dt = -v / (10*ms) : 1')
    # Just make sure that state updaters know about the two state variables
    variables = {'v': Variable(name='v', unit=1),
                 'w': Variable(name='w', unit=1)}
    
    # all methods should work for these equations.
    # First, specify them explicitly (using the object)
    for integrator in (linear, euler, exponential_euler, #TODO: Removed "independent" here due to the issue in sympy 0.7.4
                       rk2, rk4, heun, milstein):
        with catch_logs() as logs:
            returned = apply_stateupdater(eqs, variables,
                                          method=integrator)
            assert len(logs) == 0, 'Got %d unexpected warnings: %s' % (len(logs), str([l[2] for l in logs]))
    
    # Equation with multiplicative noise, only milstein and heun should work
    eqs = Equations('dv/dt = -v / (10*ms) + v*xi*second**-.5: 1')
    for integrator in (linear, independent, euler, exponential_euler, rk2, rk4):
        assert_raises(UnsupportedEquationsException, lambda: apply_stateupdater(eqs, variables, integrator))

    for integrator in (heun, milstein):
        with catch_logs() as logs:
            returned = apply_stateupdater(eqs, variables,
                                          method=integrator)
            assert len(logs) == 0, 'Got %d unexpected warnings: %s' % (len(logs), str([l[2] for l in logs]))
    
    # Arbitrary functions (converting equations into abstract code) should
    # always work
    my_stateupdater = lambda eqs, vars: 'x_new = x'
    with catch_logs() as logs:
        returned = apply_stateupdater(eqs, variables, method=my_stateupdater)
        # No warning here
        assert len(logs) == 0
    
    
    # Specification with names
    eqs = Equations('dv/dt = -v / (10*ms) : 1')
    for name, integrator in [('linear', linear), ('euler', euler),
                             #('independent', independent), #TODO: Removed "independent" here due to the issue in sympy 0.7.4
                             ('exponential_euler', exponential_euler),
                             ('rk2', rk2), ('rk4', rk4),
                             ('heun', heun), ('milstein', milstein)]:
        with catch_logs() as logs:
            returned = apply_stateupdater(eqs, variables,
                                              method=name)
            # No warning here
            assert len(logs) == 0

    # Now all except heun and milstein should refuse to work
    eqs = Equations('dv/dt = -v / (10*ms) + v*xi*second**-.5: 1')
    for name in ['linear', 'independent', 'euler', 'exponential_euler',
                 'rk2', 'rk4']:
        assert_raises(UnsupportedEquationsException, lambda: apply_stateupdater(eqs,
                                                                                variables,
                                                                                method=name))

    # milstein should work
    with catch_logs() as logs:
        apply_stateupdater(eqs, variables, method='milstein')
        assert len(logs) == 0
        
    # heun should work
    with catch_logs() as logs:
        apply_stateupdater(eqs, variables, method='heun')
        assert len(logs) == 0
    
    # non-existing name
    assert_raises(ValueError, lambda: apply_stateupdater(eqs,
                                                         variables,
                                                         method='does_not_exist'))
    
    # Automatic state updater choice should return linear for linear equations,
    # euler for non-linear, non-stochastic equations and equations with
    # additive noise, heun for equations with multiplicative noise
    # Because it is somewhat fragile, the "independent" state updater is not
    # included in this list
    all_methods = ['linear', 'exponential_euler', 'euler', 'heun', 'milstein']
    eqs = Equations('dv/dt = -v / (10*ms) : 1')
    with catch_logs(log_level=logging.INFO) as logs:
        apply_stateupdater(eqs, variables, all_methods)
        assert len(logs) == 1
        assert 'linear' in logs[0][2]
    
    # This is conditionally linear
    eqs = Equations('''dv/dt = -(v + w**2)/ (10*ms) : 1
                       dw/dt = -w/ (10*ms) : 1''')
    with catch_logs(log_level=logging.INFO) as logs:
        apply_stateupdater(eqs, variables, all_methods)
        assert len(logs) == 1
        assert 'exponential_euler' in logs[0][2]

    # # Do not test for now
    # eqs = Equations('dv/dt = sin(t) / (10*ms) : 1')
    # assert apply_stateupdater(eqs, variables) is independent

    eqs = Equations('dv/dt = -sqrt(v) / (10*ms) : 1')
    with catch_logs(log_level=logging.INFO) as logs:
        apply_stateupdater(eqs, variables, all_methods)
        assert len(logs) == 1
        assert "'euler'" in logs[0][2]

    eqs = Equations('dv/dt = -v / (10*ms) + 0.1*second**-.5*xi: 1')
    with catch_logs(log_level=logging.INFO) as logs:
        apply_stateupdater(eqs, variables, all_methods)
        assert len(logs) == 1
        assert "'euler'" in logs[0][2]

    eqs = Equations('dv/dt = -v / (10*ms) + v*0.1*second**-.5*xi: 1')
    with catch_logs(log_level=logging.INFO) as logs:
        apply_stateupdater(eqs, variables, all_methods)
        assert len(logs) == 1
        assert "'heun'" in logs[0][2]

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_subexpressions_basic():
    '''
    Make sure that the integration of a (non-stochastic) differential equation
    does not depend on whether it's formulated using subexpressions.
    '''
    # no subexpression
    eqs1 = 'dv/dt = (-v + sin(2*pi*100*Hz*t)) / (10*ms) : 1'
    # same with subexpression
    eqs2 = '''dv/dt = I / (10*ms) : 1
              I = -v + sin(2*pi*100*Hz*t): 1'''
    method = 'euler'
    G1 = NeuronGroup(1, eqs1, method=method)
    G1.v = 1
    G2 = NeuronGroup(1, eqs2, method=method)
    G2.v = 1
    mon1 = StateMonitor(G1, 'v', record=True)
    mon2 = StateMonitor(G2, 'v', record=True)
    run(10*ms)
    assert_equal(mon1.v, mon2.v, 'Results for method %s differed!' % method)


def test_subexpressions():
    '''
    Make sure that the integration of a (non-stochastic) differential equation
    does not depend on whether it's formulated using subexpressions.
    '''
    # no subexpression
    eqs1 = 'dv/dt = (-v + sin(2*pi*100*Hz*t)) / (10*ms) : 1'
    # same with subexpression
    eqs2 = '''dv/dt = I / (10*ms) : 1
              I = -v + sin(2*pi*100*Hz*t): 1'''
    
    methods = ['exponential_euler', 'rk2', 'rk4']  # euler is tested in test_subexpressions_basic
    for method in methods:
        G1 = NeuronGroup(1, eqs1, method=method)
        G1.v = 1
        G2 = NeuronGroup(1, eqs2, method=method)
        G2.v = 1
        mon1 = StateMonitor(G1, 'v', record=True)
        mon2 = StateMonitor(G2, 'v', record=True)
        net = Network(G1, mon1, G2, mon2)
        net.run(10*ms)
        assert_equal(mon1.v, mon2.v, 'Results for method %s differed!' % method)


@attr('codegen-independent')
def test_locally_constant_check():
    default_dt = defaultclock.dt
    # The linear state update can handle additive time-dependent functions
    # (e.g. a TimedArray) but only if it can be safely assumed that the function
    # is constant over a single time check
    ta0 = TimedArray(np.array([1]), dt=default_dt)  # ok
    ta1 = TimedArray(np.array([1]), dt=2*default_dt)  # ok
    ta2 = TimedArray(np.array([1]), dt=default_dt/2)  # not ok
    ta3 = TimedArray(np.array([1]), dt=default_dt*1.5)  # not ok

    for ta_func, ok in zip([ta0, ta1, ta2, ta3], [True, True, False, False]):
        # additive
        G = NeuronGroup(1, 'dv/dt = -v/(10*ms) + ta(t)*Hz : 1',
                        method='linear', namespace={'ta': ta_func})
        net = Network(G)
        if ok:
            # This should work
            net.run(0*ms)
        else:
            # This should not
            with catch_logs():
                assert_raises(UnsupportedEquationsException, lambda: net.run(0*ms))

        # multiplicative
        G = NeuronGroup(1, 'dv/dt = -v*ta(t)/(10*ms) : 1',
                        method='linear', namespace={'ta': ta_func})
        net = Network(G)
        if ok:
            # This should work
            net.run(0*ms)
        else:
            # This should not
            with catch_logs():
                assert_raises(UnsupportedEquationsException, lambda: net.run(0*ms))

    # If the argument is more than just "t", we cannot guarantee that it is
    # actually locally constant
    G = NeuronGroup(1, 'dv/dt = -v*ta(t/2.0)/(10*ms) : 1',
                        method='linear', namespace={'ta': ta0})
    net = Network(G)
    assert_raises(UnsupportedEquationsException, lambda: net.run(0*ms))

    # Arbitrary functions are not constant over a time step
    G = NeuronGroup(1, 'dv/dt = -v/(10*ms) + sin(2*pi*100*Hz*t)*Hz : 1',
                    method='linear')
    net = Network(G)
    assert_raises(UnsupportedEquationsException, lambda: net.run(0*ms))

    # Neither is "t" itself
    G = NeuronGroup(1, 'dv/dt = -v/(10*ms) + t/second**2 : 1', method='linear')
    net = Network(G)
    assert_raises(UnsupportedEquationsException, lambda: net.run(0*ms))

    # But if the argument is not referring to t, all should be well
    G = NeuronGroup(1, 'dv/dt = -v/(10*ms) + sin(2*pi*100*Hz*5*second)*Hz : 1',
                    method='linear')
    net = Network(G)
    net.run(0*ms)

def test_refractory():
    # Compare integration with and without the addition of refractoriness --
    # note that the cell here is not spiking, so it should never be in the
    # refractory period and therefore the results should be exactly identical
    # with and without (unless refractory)
    eqs_base = 'dv/dt = -v/(10*ms) : 1'
    for method in ['linear', 'independent', 'euler', 'exponential_euler', 'rk2', 'rk4']:
        G_no_ref = NeuronGroup(10, eqs_base, method=method)
        G_no_ref.v = '(i+1)/11.'
        G_ref = NeuronGroup(10, eqs_base + '(unless refractory)',
                            refractory=1*ms, method=method)
        G_ref.v = '(i+1)/11.'
        net = Network(G_ref, G_no_ref)
        net.run(10*ms)
        assert_allclose(G_no_ref.v[:], G_ref.v[:],
                        err_msg=('Results with and without refractoriness '
                                 'differ for method %s.') % method)

@with_setup(setup=store_randn, teardown=restore_randn)
def test_refractory_stochastic():
    # Fake stochastictiy, the random number generator always returns 0.5
    DEFAULT_FUNCTIONS['randn'] = fake_randn

    eqs_base = 'dv/dt = -v/(10*ms) + second**-.5*xi : 1'

    for method in ['euler', 'heun', 'milstein']:
        G_no_ref = NeuronGroup(10, eqs_base, method=method)
        G_no_ref.v = '(i+1)/11.'
        G_ref = NeuronGroup(10, eqs_base + ' (unless refractory)',
                            refractory=1*ms, method=method)
        G_ref.v = '(i+1)/11.'
        net = Network(G_ref, G_no_ref)
        net.run(10*ms)
        assert_allclose(G_no_ref.v[:], G_ref.v[:],
                        err_msg=('Results with and without refractoriness '
                                 'differ for method %s.') % method)

if __name__ == '__main__':
    from brian2 import prefs
    # prefs.codegen.target = 'cython'
    import time
    start = time.time()

    test_determination()
    test_explicit_stateupdater_parsing()
    test_non_autonomous_equations()
    test_str_repr()
    test_multiple_noise_variables_basic()
    test_multiple_noise_variables_extended()
    store_randn()
    test_multiple_noise_variables_deterministic_noise()
    restore_randn()
    store_randn()
    test_pure_noise_deterministic()
    restore_randn()
    test_temporary_variables()
    test_temporary_variables2()
    test_integrator_code()
    test_integrator_code2()
    test_illegal_calls()
    test_priority()
    test_registration()
    test_subexpressions()
    test_locally_constant_check()
    test_refractory()
    store_randn()
    test_refractory_stochastic()
    restore_randn()
    print 'Tests took', time.time()-start
