import uuid

import sympy
import numpy as np
from numpy.testing.utils import assert_raises, assert_equal, assert_allclose
from nose import SkipTest, with_setup
from nose.plugins.attrib import attr

from brian2.core.variables import linked_var
from brian2.core.network import Network
from brian2.core.preferences import prefs
from brian2.core.clocks import defaultclock
from brian2.devices.device import restore_device
from brian2.equations.equations import Equations
from brian2.groups.group import get_dtype
from brian2.groups.neurongroup import NeuronGroup
from brian2.synapses.synapses import Synapses
from brian2.monitors.statemonitor import StateMonitor
from brian2.units.fundamentalunits import (DimensionMismatchError,
                                           have_same_dimensions)
from brian2.units.allunits import second, volt
from brian2.units.stdunits import ms, mV, Hz
from brian2.utils.logger import catch_logs


@attr('codegen-independent')
def test_creation():
    '''
    A basic test that creating a NeuronGroup works.
    '''
    G = NeuronGroup(42, model='dv/dt = -v/(10*ms) : 1', reset='v=0',
                    threshold='v>1')
    assert len(G) == 42
        
    # Test some error conditions
    # --------------------------
    
    # Model equations as first argument (no number of neurons)
    assert_raises(TypeError, lambda: NeuronGroup('dv/dt = 5*Hz : 1', 1))
    
    # Not a number as first argument
    assert_raises(TypeError, lambda: NeuronGroup(object(), 'dv/dt = 5*Hz : 1'))
    
    # Illegal number
    assert_raises(ValueError, lambda: NeuronGroup(0, 'dv/dt = 5*Hz : 1'))
    
    # neither string nor Equations object as model description
    assert_raises(TypeError, lambda: NeuronGroup(1, object()))


@attr('codegen-independent')
def test_variables():
    '''
    Test the correct creation of the variables dictionary.
    '''
    G = NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1')
    assert 'v' in G.variables and 't' in G.variables and 'dt' in G.variables
    assert 'not_refractory' not in G.variables and 'lastspike' not in G.variables

    G = NeuronGroup(1, 'dv/dt = -v/tau + xi*tau**-0.5: 1')
    assert not 'tau' in G.variables and 'xi' in G.variables

    # NeuronGroup with refractoriness
    G = NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1', refractory=5*ms)
    assert 'not_refractory' in G.variables and 'lastspike' in G.variables

@attr('standalone-compatible')
@with_setup(teardown=restore_device)
def test_stochastic_variable():
    '''
    Test that a NeuronGroup with a stochastic variable can be simulated. Only
    makes sure no error occurs.
    '''
    tau = 10 * ms
    G = NeuronGroup(1, 'dv/dt = -v/tau + xi*tau**-0.5: 1')
    net = Network(G)
    net.run(defaultclock.dt)

@attr('standalone-compatible')
@with_setup(teardown=restore_device)
def test_stochastic_variable_multiplicative():
    '''
    Test that a NeuronGroup with multiplicative noise can be simulated. Only
    makes sure no error occurs.
    '''
    mu = 0.5/second # drift
    sigma = 0.1/second #diffusion
    G = NeuronGroup(1, 'dX/dt = (mu - 0.5*second*sigma**2)*X + X*sigma*xi*second**.5: 1')
    net = Network(G)
    net.run(defaultclock.dt)

def test_scalar_variable():
    '''
    Test the correct handling of scalar variables
    '''
    tau = 10*ms
    G = NeuronGroup(10, '''E_L : volt (shared)
                           s2 : 1 (shared)
                           dv/dt = (E_L - v) / tau : volt''')
    # Setting should work in these ways
    G.E_L = -70*mV
    assert_allclose(G.E_L[:], -70*mV)
    G.E_L[:] = -60*mV
    assert_allclose(G.E_L[:], -60*mV)
    G.E_L = 'E_L + s2*mV - 10*mV'
    assert_allclose(G.E_L[:], -70*mV)
    G.E_L[:] = '-75*mV'
    assert_allclose(G.E_L[:], -75*mV)
    net = Network(G)
    net.run(defaultclock.dt)

@attr('standalone-compatible')
@with_setup(teardown=restore_device)
def test_referred_scalar_variable():
    '''
    Test the correct handling of referred scalar variables in subexpressions
    '''
    G = NeuronGroup(10, '''out = sin(2*pi*t*freq) + x: 1
                           x : 1
                           freq : Hz (shared)''')
    G.freq = 1*Hz
    G.x = np.arange(10)
    G2 = NeuronGroup(10, '')
    G2.variables.add_reference('out', G)
    net = Network(G, G2)
    net.run(.25*second)
    assert_allclose(G2.out[:], np.arange(10)+1)

@attr('standalone-compatible')
@with_setup(teardown=restore_device)
def test_linked_variable_correct():
    '''
    Test correct uses of linked variables.
    '''
    tau = 10*ms
    G1 = NeuronGroup(10, 'dv/dt = -v / tau : volt')
    G1.v = np.linspace(0*mV, 20*mV, 10)
    G2 = NeuronGroup(10, 'v : volt (linked)')
    G2.v = linked_var(G1.v)
    mon1 = StateMonitor(G1, 'v', record=True)
    mon2 = StateMonitor(G2, 'v', record=True)
    net = Network(G1, G2, mon1, mon2)
    net.run(10*ms)
    assert_equal(mon1.v[:, :], mon2.v[:, :])
    # Make sure that printing the variable values works
    assert len(str(G2.v)) > 0
    assert len(repr(G2.v)) > 0
    assert len(str(G2.v[:])) > 0
    assert len(repr(G2.v[:])) > 0

@attr('codegen-independent')
def test_linked_variable_incorrect():
    '''
    Test incorrect uses of linked variables.
    '''
    G1 = NeuronGroup(10, '''x : volt
                            y : 1''')
    G2 = NeuronGroup(20, '''x: volt''')
    G3 = NeuronGroup(10, '''l : volt (linked)
                            not_linked : volt''')

    # incorrect unit
    assert_raises(DimensionMismatchError, lambda: setattr(G3, 'l', linked_var(G1.y)))
    # incorrect group size
    assert_raises(ValueError, lambda: setattr(G3, 'l', linked_var(G2.x)))
    # incorrect use of linked_var
    assert_raises(ValueError, lambda: setattr(G3, 'l', linked_var(G1.x, 'x')))
    assert_raises(ValueError, lambda: setattr(G3, 'l', linked_var(G1)))
    # Not a linked variable
    assert_raises(TypeError, lambda: setattr(G3, 'not_linked', linked_var(G1.x)))

@attr('standalone-compatible')
@with_setup(teardown=restore_device)
def test_linked_variable_scalar():
    '''
    Test linked variable from a size 1 group.
    '''
    G1 = NeuronGroup(1, 'dx/dt = -x / (10*ms) : 1')
    G2 = NeuronGroup(10, '''dy/dt = (-y + x) / (20*ms) : 1
                            x : 1 (linked)''')
    G1.x = 1
    G2.y = np.linspace(0, 1, 10)
    G2.x = linked_var(G1.x)
    mon = StateMonitor(G2, 'y', record=True)
    net = Network(G1, G2, mon)
    # We don't test anything for now, except that it runs without raising an
    # error
    net.run(10*ms)
    # Make sure that printing the variable values works
    assert len(str(G2.x)) > 0
    assert len(repr(G2.x)) > 0
    assert len(str(G2.x[:])) > 0
    assert len(repr(G2.x[:])) > 0

@attr('codegen-independent')
def test_linked_variable_indexed():
    '''
    Test linking a variable with an index specified as an array
    '''
    G = NeuronGroup(10, '''x : 1
                           y : 1 (linked)''')

    G.x = np.arange(10)*0.1
    G.y = linked_var(G.x, index=np.arange(10)[::-1])
    # G.y should refer to an inverted version of G.x
    assert_equal(G.y[:], np.arange(10)[::-1]*0.1)

@attr('codegen-independent')
def test_linked_variable_repeat():
    '''
    Test a "repeat"-like connection between two groups of different size
    '''
    G1 = NeuronGroup(5, 'w : 1')
    G2 = NeuronGroup(10, 'v : 1 (linked)')
    G2.v = linked_var(G1.w, index=np.arange(5).repeat(2))
    G1.w = np.arange(5) * 0.1
    assert_equal(G2.v[:], np.arange(5).repeat(2) * 0.1)

@attr('codegen-independent')
def test_linked_double_linked1():
    '''
    Linked to a linked variable, without indices
    '''
    G1 = NeuronGroup(10, 'x : 1')
    G2 = NeuronGroup(10, 'y : 1 (linked)')
    G2.y = linked_var(G1.x)
    G3 = NeuronGroup(10, 'z: 1 (linked)')
    G3.z = linked_var(G2.y)

    G1.x = np.arange(10)
    assert_equal(G3.z[:], np.arange(10))

@attr('codegen-independent')
def test_linked_double_linked2():
    '''
    Linked to a linked variable, first without indices, second with indices
    '''

    G1 = NeuronGroup(5, 'x : 1')
    G2 = NeuronGroup(5, 'y : 1 (linked)')
    G2.y = linked_var(G1.x)
    G3 = NeuronGroup(10, 'z: 1 (linked)')
    G3.z = linked_var(G2.y, index=np.arange(5).repeat(2))

    G1.x = np.arange(5)*0.1
    assert_equal(G3.z[:], np.arange(5).repeat(2)*0.1)

@attr('codegen-independent')
def test_linked_double_linked3():
    '''
    Linked to a linked variable, first with indices, second without indices
    '''
    G1 = NeuronGroup(5, 'x : 1')
    G2 = NeuronGroup(10, 'y : 1 (linked)')
    G2.y = linked_var(G1.x, index=np.arange(5).repeat(2))
    G3 = NeuronGroup(10, 'z: 1 (linked)')
    G3.z = linked_var(G2.y)

    G1.x = np.arange(5)*0.1
    assert_equal(G3.z[:], np.arange(5).repeat(2)*0.1)

@attr('codegen-independent')
def test_linked_double_linked4():
    '''
    Linked to a linked variable, both use indices
    '''
    G1 = NeuronGroup(5, 'x : 1')
    G2 = NeuronGroup(10, 'y : 1 (linked)')
    G2.y = linked_var(G1.x, index=np.arange(5).repeat(2))
    G3 = NeuronGroup(10, 'z: 1 (linked)')
    G3.z = linked_var(G2.y, index=np.arange(10)[::-1])

    G1.x = np.arange(5)*0.1
    assert_equal(G3.z[:], np.arange(5).repeat(2)[::-1]*0.1)

@attr('codegen-independent')
def test_linked_triple_linked():
    '''
    Link to a linked variable that links to a linked variable, all use indices
    '''
    G1 = NeuronGroup(2, 'a : 1')

    G2 = NeuronGroup(4, 'b : 1 (linked)')
    G2.b = linked_var(G1.a, index=np.arange(2).repeat(2))

    G3 = NeuronGroup(4, 'c: 1 (linked)')
    G3.c = linked_var(G2.b, index=np.arange(4)[::-1])

    G4 = NeuronGroup(8, 'd: 1 (linked)')
    G4.d = linked_var(G3.c, index=np.arange(4).repeat(2))

    G1.a = np.arange(2)*0.1
    assert_equal(G4.d[:], np.arange(2).repeat(2)[::-1].repeat(2)*0.1)

@attr('codegen-independent')
def test_linked_subgroup():
    '''
    Test linking a variable from a subgroup
    '''
    G1 = NeuronGroup(10, 'x : 1')
    G1.x = np.arange(10) * 0.1
    G2 = G1[3:8]
    G3 = NeuronGroup(5, 'y:1 (linked)')
    G3.y = linked_var(G2.x)

    assert_equal(G3.y[:], (np.arange(5)+3)*0.1)

@attr('codegen-independent')
def test_linked_subgroup2():
    '''
    Test linking a variable from a subgroup with indexing
    '''
    G1 = NeuronGroup(10, 'x : 1')
    G1.x = np.arange(10) * 0.1
    G2 = G1[3:8]
    G3 = NeuronGroup(10, 'y:1 (linked)')
    G3.y = linked_var(G2.x, index=np.arange(5).repeat(2))

    assert_equal(G3.y[:], (np.arange(5)+3).repeat(2)*0.1)

@attr('standalone-compatible')
@with_setup(teardown=restore_device)
def test_linked_subexpression():
    '''
    Test a subexpression referring to a linked variable.
    '''
    G = NeuronGroup(2, 'dv/dt = 100*Hz : 1',
                    threshold='v>1', reset='v=0')
    G.v = [0, .5]
    G2 = NeuronGroup(10, '''I = clip(x, 0, inf) : 1
                            x : 1 (linked) ''')

    G2.x = linked_var(G.v, index=np.array([0, 1]).repeat(5))
    mon = StateMonitor(G2, 'I', record=True)

    net = Network(G, G2, mon)
    net.run(5*ms)

    # Due to the linking, the first 5 and the second 5 recorded I vectors should
    # be identical
    assert all((all(mon[i].I == mon[0].I) for i in xrange(5)))
    assert all((all(mon[i+5].I == mon[5].I) for i in xrange(5)))

@attr('standalone-compatible')
@with_setup(teardown=restore_device)
def test_linked_subexpression_2():
    '''
    Test a linked variable referring to a subexpression without indices
    '''
    G = NeuronGroup(2, '''dv/dt = 100*Hz : 1
                          I = clip(v, 0, inf) : 1''',
                    threshold='v>1', reset='v=0')
    G.v = [0, .5]
    G2 = NeuronGroup(2, '''I_l : 1 (linked) ''')

    G2.I_l = linked_var(G.I)
    mon1 = StateMonitor(G, 'I', record=True)
    mon = StateMonitor(G2, 'I_l', record=True)

    net = Network(G, G2, mon, mon1)
    net.run(5*ms)

    assert all(mon[0].I_l == mon1[0].I)
    assert all(mon[1].I_l == mon1[1].I)

@attr('standalone-compatible')
@with_setup(teardown=restore_device)
def test_linked_subexpression_3():
    '''
    Test a linked variable referring to a subexpression with indices
    '''
    G = NeuronGroup(2, '''dv/dt = 100*Hz : 1
                          I = clip(v, 0, inf) : 1''',
                    threshold='v>1', reset='v=0')
    G.v = [0, .5]
    G2 = NeuronGroup(10, '''I_l : 1 (linked) ''')

    G2.I_l = linked_var(G.I, index=np.array([0, 1]).repeat(5))
    mon1 = StateMonitor(G, 'I', record=True)
    mon = StateMonitor(G2, 'I_l', record=True)

    net = Network(G, G2, mon, mon1)
    net.run(5*ms)

    # Due to the linking, the first 5 and the second 5 recorded I vectors should
    # refer to the
    assert all((all(mon[i].I_l == mon1[0].I) for i in xrange(5)))
    assert all((all(mon[i+5].I_l == mon1[1].I) for i in xrange(5)))


def test_linked_subexpression_synapse():
    '''
    Test a complicated setup (not unlikely when using brian hears)
    '''
    G = NeuronGroup(2, 'dv/dt = 100*Hz : 1',
                    threshold='v>1', reset='v=0')
    G.v = [0, .5]
    G2 = NeuronGroup(10, '''I = clip(x, 0, inf) : 1
                            x : 1 (linked) ''')

    # This will not be able to include references to `I` as `I_pre` etc., since
    # the indirect indexing would have to change depending on the synapses
    G2.x = linked_var(G.v, index=np.array([0, 1]).repeat(5))
    S = Synapses(G2, G2, '')
    S.connect('i==j')
    assert 'I' not in S.variables
    assert 'I_pre' not in S.variables
    assert 'I_post' not in S.variables
    assert 'x' not in S.variables
    assert 'x_pre' not in S.variables
    assert 'x_post' not in S.variables


@attr('codegen-independent')
def test_linked_variable_indexed_incorrect():
    '''
    Test errors when providing incorrect index arrays
    '''
    G = NeuronGroup(10, '''x : 1
                           y : 1 (linked)''')

    G.x = np.arange(10)*0.1
    assert_raises(TypeError,
                  lambda: setattr(G, 'y',
                                  linked_var(G.x, index=np.arange(10)*1.0)))
    assert_raises(TypeError,
                  lambda: setattr(G, 'y',
                                  linked_var(G.x, index=np.arange(10).reshape(5, 2))))
    assert_raises(TypeError,
                  lambda: setattr(G, 'y',
                                  linked_var(G.x, index=np.arange(5))))
    assert_raises(ValueError,
                  lambda: setattr(G, 'y',
                                  linked_var(G.x, index=np.arange(10)-1)))
    assert_raises(ValueError,
                  lambda: setattr(G, 'y',
                                  linked_var(G.x, index=np.arange(10)+1)))

@attr('codegen-independent')
def test_linked_synapses():
    '''
    Test linking to a synaptic variable (should raise an error).
    '''
    G = NeuronGroup(10, '')
    S = Synapses(G, G, 'w:1', connect=True)
    G2 = NeuronGroup(100, 'x : 1 (linked)')
    assert_raises(NotImplementedError, lambda: setattr(G2, 'x', linked_var(S, 'w')))

@attr('standalone-compatible')
@with_setup(teardown=restore_device)
def test_linked_var_in_reset():
    G1 = NeuronGroup(3, 'x:1')
    G2 = NeuronGroup(3, '''x_linked : 1 (linked)
                           y:1''',
                     threshold='y>1', reset='y=0; x_linked += 1')
    G2.x_linked = linked_var(G1, 'x')
    G2.y = [0, 1.1, 0]
    net = Network(G1, G2)
    # In this context, x_linked should not be considered as a scalar variable
    # and therefore the reset statement should be allowed
    net.run(3*defaultclock.dt)
    assert_equal(G1.x[:], [0, 1, 0])

@attr('standalone-compatible')
@with_setup(teardown=restore_device)
def test_linked_var_in_reset_size_1():
    G1 = NeuronGroup(1, 'x:1')
    G2 = NeuronGroup(1, '''x_linked : 1 (linked)
                           y:1''',
                     threshold='y>1', reset='y=0; x_linked += 1')
    G2.x_linked = linked_var(G1, 'x')
    G2.y = 1.1
    net = Network(G1, G2)
    # In this context, x_linked should not be considered as a scalar variable
    # and therefore the reset statement should be allowed
    net.run(3*defaultclock.dt)
    assert_equal(G1.x[:], 1)

@attr('codegen-independent')
def test_linked_var_in_reset_incorrect():
    # Raise an error if a scalar variable (linked variable from a group of size
    # 1 is set in a reset statement of a group with size > 1)
    G1 = NeuronGroup(1, 'x:1')
    G2 = NeuronGroup(2, '''x_linked : 1 (linked)
                           y:1''',
                     threshold='y>1', reset='y=0; x_linked += 1')
    G2.x_linked = linked_var(G1, 'x')
    G2.y = 1.1
    net = Network(G1, G2)
    # It is not well-defined what x_linked +=1 means in this context
    # (as for any other shared variable)
    assert_raises(SyntaxError, lambda: net.run(0*ms))

@attr('codegen-independent')
def test_unit_errors():
    '''
    Test that units are checked for a complete namespace.
    '''
    # Unit error in model equations
    assert_raises(DimensionMismatchError,
                  lambda: NeuronGroup(1, 'dv/dt = -v : 1'))
    assert_raises(DimensionMismatchError,
                  lambda: NeuronGroup(1, 'dv/dt = -v/(10*ms) + 2*mV: 1'))

@attr('codegen-independent')
def test_incomplete_namespace():
    '''
    Test that the namespace does not have to be complete at creation time.
    '''
    # This uses tau which is not defined yet (explicit namespace)
    G = NeuronGroup(1, 'dv/dt = -v/tau : 1', namespace={})
    G.namespace['tau'] = 10*ms
    net = Network(G)
    net.run(0*ms)

    # This uses tau which is not defined yet (implicit namespace)
    G = NeuronGroup(1, 'dv/dt = -v/tau : 1')
    tau = 10*ms
    net = Network(G)
    net.run(0*ms)

@attr('codegen-independent')
def test_namespace_errors():

    # model equations use unknown identifier
    G = NeuronGroup(1, 'dv/dt = -v/tau : 1')
    net = Network(G)
    assert_raises(KeyError, lambda: net.run(1*ms))

    # reset uses unknown identifier
    G = NeuronGroup(1, 'dv/dt = -v/tau : 1', reset='v = v_r')
    net = Network(G)
    assert_raises(KeyError, lambda: net.run(1*ms))

    # threshold uses unknown identifier
    G = NeuronGroup(1, 'dv/dt = -v/tau : 1', threshold='v > v_th')
    net = Network(G)
    assert_raises(KeyError, lambda: net.run(1*ms))

@attr('codegen-independent')
def test_namespace_warnings():
    G = NeuronGroup(1, '''x : 1
                          y : 1''',
                    # unique names to get warnings every time:
                    name='neurongroup_'+str(uuid.uuid4()).replace('-', '_'))
    # conflicting variable in namespace
    y = 5
    with catch_logs() as l:
        G.x = 'y'
        assert len(l) == 1, 'got %s as warnings' % str(l)
        assert l[0][1].endswith('.resolution_conflict')

    del y

    # conflicting variables with special meaning
    i = 5
    N = 3
    with catch_logs() as l:
        G.x = 'i / N'
        assert len(l) == 2, 'got %s as warnings' % str(l)
        assert l[0][1].endswith('.resolution_conflict')
        assert l[1][1].endswith('.resolution_conflict')

    del i
    del N
    # conflicting variables in equations
    y = 5*Hz
    G = NeuronGroup(1, '''y : Hz
                          dx/dt = y : 1''',
                    # unique names to get warnings every time:
                    name='neurongroup_'+str(uuid.uuid4()).replace('-', '_'))

    net = Network(G)
    with catch_logs() as l:
        net.run(0*ms)
        assert len(l) == 1, 'got %s as warnings' % str(l)
        assert l[0][1].endswith('.resolution_conflict')
    del y

    i = 5
    # i is referring to the neuron number:
    G = NeuronGroup(1, '''dx/dt = i*Hz : 1''',
                    # unique names to get warnings every time:
                    name='neurongroup_'+str(uuid.uuid4()).replace('-', '_'))
    net = Network(G)
    with catch_logs() as l:
        net.run(0*ms)
        assert len(l) == 1, 'got %s as warnings' % str(l)
        assert l[0][1].endswith('.resolution_conflict')
    del i

    # Variables that are used internally but not in equations should not raise
    # a warning
    N = 3
    i = 5
    dt = 1*ms
    G = NeuronGroup(1, '''dx/dt = x/(10*ms) : 1''',
                    # unique names to get warnings every time:
                    name='neurongroup_'+str(uuid.uuid4()).replace('-', '_'))
    net = Network(G)
    with catch_logs() as l:
        net.run(0*ms)
        assert len(l) == 0, 'got %s as warnings' % str(l)

@attr('standalone-compatible')
@with_setup(teardown=restore_device)
def test_threshold_reset():
    '''
    Test that threshold and reset work in the expected way.
    '''
    # Membrane potential does not change by itself
    G = NeuronGroup(3, 'dv/dt = 0 / second : 1',
                    threshold='v > 1', reset='v=0.5')
    G.v = np.array([0, 1, 2])
    net = Network(G)
    net.run(defaultclock.dt)
    assert_equal(G.v[:], np.array([0, 1, 0.5]))

@attr('codegen-independent')
def test_unit_errors_threshold_reset():
    '''
    Test that unit errors in thresholds and resets are detected.
    '''
    # Unit error in threshold
    assert_raises(DimensionMismatchError,
                  lambda: NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                                      threshold='v > -20*mV'))

    # Unit error in reset
    assert_raises(DimensionMismatchError,
                  lambda: NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                                      reset='v = -65*mV'))

    # More complicated unit reset with an intermediate variable
    # This should pass
    NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                reset='''temp_var = -65
                         v = temp_var''')
    # throw in an empty line (should still pass)
    NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                reset='''temp_var = -65

                         v = temp_var''')

    # This should fail
    assert_raises(DimensionMismatchError,
                  lambda: NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                                      reset='''temp_var = -65*mV
                                               v = temp_var'''))

    # Resets with an in-place modification
    # This should work
    NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                reset='''v /= 2''')

    # This should fail
    assert_raises(DimensionMismatchError,
                  lambda: NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                                      reset='''v -= 60*mV'''))

@attr('codegen-independent')
def test_syntax_errors():
    '''
    Test that syntax errors are already caught at initialization time.
    For equations this is already tested in test_equations
    '''
    
    # We do not specify the exact type of exception here: Python throws a
    # SyntaxError while C++ results in a ValueError
    # Syntax error in threshold
    assert_raises(Exception,
                  lambda: NeuronGroup(1, 'dv/dt = 5*Hz : 1',
                                      threshold='>1'),
                  )

    # Syntax error in reset
    assert_raises(Exception,
                  lambda: NeuronGroup(1, 'dv/dt = 5*Hz : 1',
                                      reset='0'))


def test_state_variables():
    '''
    Test the setting and accessing of state variables.
    '''
    G = NeuronGroup(10, 'v : volt')

    # The variable N should be always present
    assert G.N == 10
    # But it should be read-only
    assert_raises(TypeError, lambda: G.__setattr__('N', 20))
    assert_raises(TypeError, lambda: G.__setattr__('N_', 20))

    G.v = -70*mV
    assert_raises(DimensionMismatchError, lambda: G.__setattr__('v', -70))
    G.v_ = float(-70*mV)
    assert_allclose(G.v[:], -70*mV)
    G.v = -70*mV + np.arange(10)*mV
    assert_allclose(G.v[:], -70*mV + np.arange(10)*mV)
    G.v = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * volt
    assert_allclose(G.v[:], np.arange(10) * volt)
    # incorrect size
    assert_raises(ValueError, lambda: G.__setattr__('v', [0, 1]*volt))
    assert_raises(ValueError, lambda: G.__setattr__('v', np.arange(11)*volt))

    G.v = -70*mV
    # Numpy methods should be able to deal with state variables
    # (discarding units)
    assert_allclose(np.mean(G.v), float(-70*mV))
    # Getting the content should return a Quantity object which then natively
    # supports numpy functions that access a method
    assert_allclose(np.mean(G.v[:]), -70*mV)

    # You should also be able to set variables with a string
    G.v = '-70*mV + i*mV'
    assert_allclose(G.v[0], -70*mV)
    assert_allclose(G.v[9], -61*mV)
    assert_allclose(G.v[:], -70*mV + np.arange(10)*mV)

    # And it should raise an unit error if the units are incorrect
    assert_raises(DimensionMismatchError,
                  lambda: G.__setattr__('v', '70 + i'))
    assert_raises(DimensionMismatchError,
                  lambda: G.__setattr__('v', '70 + i*mV'))

    # Calculating with state variables should work too
    # With units
    assert all(G.v - G.v == 0)
    assert all(G.v - G.v[:] == 0*mV)
    assert all(G.v[:] - G.v == 0*mV)
    assert all(G.v + 70*mV == G.v[:] + 70*mV)
    assert all(70*mV + G.v == G.v[:] + 70*mV)
    assert all(G.v + G.v == 2*G.v)
    assert all(G.v / 2.0 == 0.5*G.v)
    assert all(1.0 / G.v == 1.0 / G.v[:])
    assert_equal((-G.v)[:], -G.v[:])
    assert_equal((+G.v)[:], G.v[:])
    #Without units
    assert all(G.v_ - G.v_ == 0)
    assert all(G.v_ - G.v_[:] == 0)
    assert all(G.v_[:] - G.v_ == 0)
    assert all(G.v_ + float(70*mV) == G.v_[:] + float(70*mV))
    assert all(float(70*mV) + G.v_ == G.v_[:] + float(70*mV))
    assert all(G.v_ + G.v_ == 2*G.v_)
    assert all(G.v_ / 2.0 == 0.5*G.v_)
    assert all(1.0 / G.v_ == 1.0 / G.v_[:])
    assert_equal((-G.v)[:], -G.v[:])
    assert_equal((+G.v)[:], G.v[:])

    # And in-place modification should work as well
    G.v += 10*mV
    G.v -= 10*mV
    G.v *= 2
    G.v /= 2.0

    # with unit checking
    assert_raises(DimensionMismatchError, lambda: G.v.__iadd__(3*second))
    assert_raises(DimensionMismatchError, lambda: G.v.__iadd__(3))
    assert_raises(DimensionMismatchError, lambda: G.v.__imul__(3*second))

    # in-place modification with strings should not work
    assert_raises(TypeError, lambda: G.v.__iadd__('string'))
    assert_raises(TypeError, lambda: G.v.__imul__('string'))
    assert_raises(TypeError, lambda: G.v.__idiv__('string'))
    assert_raises(TypeError, lambda: G.v.__isub__('string'))


def test_state_variable_access():
    G = NeuronGroup(10, 'v:volt')
    G.v = np.arange(10) * volt

    assert_equal(np.asarray(G.v[:]), np.arange(10))
    assert have_same_dimensions(G.v[:], volt)
    assert_equal(np.asarray(G.v[:]), G.v_[:])
    # Accessing single elements, slices and arrays
    assert G.v[5] == 5 * volt
    assert G.v_[5] == 5
    assert_equal(G.v[:5], np.arange(5) * volt)
    assert_equal(G.v_[:5], np.arange(5))
    assert_equal(G.v[[0, 5]], [0, 5] * volt)
    assert_equal(G.v_[[0, 5]], np.array([0, 5]))

    # Illegal indexing
    assert_raises(IndexError, lambda: G.v[0, 0])
    assert_raises(IndexError, lambda: G.v_[0, 0])
    assert_raises(TypeError, lambda: G.v[object()])
    assert_raises(TypeError, lambda: G.v_[object()])

    # A string representation should not raise any error
    assert len(str(G.v))
    assert len(repr(G.v))
    assert len(str(G.v_))
    assert len(repr(G.v_))


def test_state_variable_access_strings():
    G = NeuronGroup(10, '''v : volt
                           dv_ref/dt = -v_ref/(10*ms) : 1 (unless refractory)''',
                    threshold='v_ref>1', reset='v_ref=1', refractory=1*ms)
    G.v = np.arange(10) * volt
    # Indexing with strings
    assert G.v['i==2'] == G.v[2]
    assert G.v_['i==2'] == G.v_[2]
    assert_equal(G.v['v >= 3*volt'], G.v[3:])
    assert_equal(G.v_['v >= 3*volt'], G.v_[3:])
    # Should also check for units
    assert_raises(DimensionMismatchError, lambda: G.v['v >= 3'])
    assert_raises(DimensionMismatchError, lambda: G.v['v >= 3*second'])

    # Setting with strings
    # --------------------
    # String value referring to i
    G.v = '2*i*volt'
    assert_equal(G.v[:], 2*np.arange(10)*volt)
    # String value referring to i
    G.v[:5] = '3*i*volt'
    assert_equal(G.v[:],
                 np.array([0, 3, 6, 9, 12, 10, 12, 14, 16, 18])*volt)

    G.v = np.arange(10) * volt

    # Conditional write variable
    G.v_ref = '2*i'
    assert_equal(G.v_ref[:], 2*np.arange(10))

    # String value referring to a state variable
    G.v = '2*v'
    assert_equal(G.v[:], 2*np.arange(10)*volt)
    G.v[:5] = '2*v'
    assert_equal(G.v[:],
                 np.array([0, 4, 8, 12, 16, 10, 12, 14, 16, 18])*volt)

    G.v = np.arange(10) * volt
    # String value referring to state variables, i, and an external variable
    ext = 5*volt
    G.v = 'v + ext + (N + i)*volt'
    assert_equal(G.v[:], 2*np.arange(10)*volt + 15*volt)

    G.v = np.arange(10) * volt
    G.v[:5] = 'v + ext + (N + i)*volt'
    assert_equal(G.v[:],
                 np.array([15, 17, 19, 21, 23, 5, 6, 7, 8, 9])*volt)

    G.v = 'v + randn()*volt'  # only check that it doesn't raise an error
    G.v[:5] = 'v + randn()*volt'  # only check that it doesn't raise an error

    G.v = np.arange(10) * volt
    # String index using a random number
    G.v['rand() <= 1'] = 0*mV
    assert_equal(G.v[:], np.zeros(10)*volt)

    G.v = np.arange(10) * volt
    # String index referring to i and setting to a scalar value
    G.v['i>=5'] = 0*mV
    assert_equal(G.v[:], np.array([0, 1, 2, 3, 4, 0, 0, 0, 0, 0])*volt)
    # String index referring to a state variable
    G.v['v<3*volt'] = 0*mV
    assert_equal(G.v[:], np.array([0, 0, 0, 3, 4, 0, 0, 0, 0, 0])*volt)
    # String index referring to state variables, i, and an external variable
    ext = 2*volt
    G.v['v>=ext and i==(N-6)'] = 0*mV
    assert_equal(G.v[:], np.array([0, 0, 0, 3, 0, 0, 0, 0, 0, 0])*volt)

    G.v = np.arange(10) * volt
    # Strings for both condition and values
    G.v['i>=5'] = 'v*2'
    assert_equal(G.v[:], np.array([0, 1, 2, 3, 4, 10, 12, 14, 16, 18])*volt)
    G.v['v>=5*volt'] = 'i*volt'
    assert_equal(G.v[:], np.arange(10)*volt)

@attr('codegen-independent')
def test_unknown_state_variables():
    # Test how setting attribute names that do not correspond to a state
    # variable are handled
    G = NeuronGroup(10, 'v : 1')
    assert_raises(AttributeError, lambda: setattr(G, 'unknown', 42))

    # Creating a new private attribute should be fine
    G._unknown = 42
    assert G._unknown == 42

    # Explicitly create the attribute
    G.add_attribute('unknown')
    G.unknown = 42
    assert G.unknown == 42

@attr('codegen-independent')
def test_subexpression():
    G = NeuronGroup(10, '''dv/dt = freq : 1
                           freq : Hz
                           array : 1
                           expr = 2*freq + array*Hz : Hz''')
    G.freq = '10*i*Hz'
    G.array = 5
    assert_equal(G.expr[:], 2*10*np.arange(10)*Hz + 5*Hz)

@attr('codegen-independent')
def test_subexpression_with_constant():
        g = 2
        G = NeuronGroup(1, '''x : 1
                              I = x*g : 1''')
        G.x = 1
        assert_equal(G.I[:], np.array([2]))
        # Subexpressions that refer to external variables are tricky, see github
        # issue #313 for details

        # Comparisons
        assert G.I == 2
        assert G.I >= 1
        assert G.I > 1
        assert G.I < 3
        assert G.I <= 3
        assert G.I != 3

        # arithmetic operations
        assert G.I + 1 == 3
        assert 1 + G.I == 3
        assert G.I * 1 == 2
        assert 1 * G.I == 2
        assert G.I - 1 == 1
        assert 3 - G.I == 1
        assert G.I / 1 == 2
        assert G.I // 1 == 2.0
        assert 1.0 / G.I == 0.5
        assert 1 // G.I == 0
        assert +G.I == 2
        assert -G.I == -2

        # other operations
        assert len(G.I) == 1

        # These will not work
        assert_raises(ValueError, lambda: np.array(G.I))
        assert_raises(ValueError, lambda: np.mean(G.I))
        # But these should
        assert_equal(np.array(G.I[:]), G.I[:])
        assert np.mean(G.I[:]) == 2

        # This will work but display a text, advising to use G.I[:] instead of
        # G.I
        assert(len(str(G.I)))
        assert(len(repr(G.I)))

@attr('codegen-independent')
def test_scalar_parameter_access():
    G = NeuronGroup(10, '''dv/dt = freq : 1
                           freq : Hz (shared)
                           number : 1 (shared)
                           array : 1''')

    # Try setting a scalar variable
    G.freq = 100*Hz
    assert_equal(G.freq[:], 100*Hz)
    G.freq[:] = 200*Hz
    assert_equal(G.freq[:], 200*Hz)
    G.freq = 'freq - 50*Hz + number*Hz'
    assert_equal(G.freq[:], 150*Hz)
    G.freq[:] = '50*Hz'
    assert_equal(G.freq[:], 50*Hz)

    # Check the second method of accessing that works
    assert_equal(np.asanyarray(G.freq), 50*Hz)

    # Check error messages
    assert_raises(IndexError, lambda: G.freq[0])
    assert_raises(IndexError, lambda: G.freq[1])
    assert_raises(IndexError, lambda: G.freq[0:1])
    assert_raises(IndexError, lambda: G.freq['i>5'])

    assert_raises(ValueError, lambda: G.freq.set_item(slice(None), [0, 1]*Hz))
    assert_raises(IndexError, lambda: G.freq.set_item(0, 100*Hz))
    assert_raises(IndexError, lambda: G.freq.set_item(1, 100*Hz))
    assert_raises(IndexError, lambda: G.freq.set_item('i>5', 100*Hz))


@attr('codegen-independent')
def test_scalar_subexpression():
    G = NeuronGroup(10, '''dv/dt = freq : 1
                           freq : Hz (shared)
                           number : 1 (shared)
                           array : 1
                           sub = freq + number*Hz : Hz (shared)''')
    G.freq = 100*Hz
    G.number = 50
    assert G.sub[:] == 150*Hz

    assert_raises(SyntaxError, lambda: NeuronGroup(10, '''dv/dt = freq : 1
                                                          freq : Hz (shared)
                                                          array : 1
                                                          sub = freq + array*Hz : Hz (shared)'''))

    # A scalar subexpresion cannot refer to implicitly vectorized functions
    assert_raises(SyntaxError, lambda: NeuronGroup(10, 'sub = rand() : 1 (shared)'))

@attr('codegen-independent')
def test_repr():
    G = NeuronGroup(10, '''dv/dt = -(v + Inp) / tau : volt
                           Inp = sin(2*pi*freq*t) : volt
                           freq : Hz''')

    # Test that string/LaTeX representations do not raise errors
    for func in [str, repr, sympy.latex]:
        assert len(func(G))
        assert len(func(G.equations))
        for eq in G.equations.itervalues():
            assert len(func(eq))

@attr('codegen-independent')
def test_indices():
    G = NeuronGroup(10, 'v : 1')
    G.v = 'i'
    ext_var = 5
    assert_equal(G.indices[:], G.i[:])
    assert_equal(G.indices[5:], G.indices['i >= 5'])
    assert_equal(G.indices[5:], G.indices['i >= ext_var'])
    assert_equal(G.indices['v >= 5'], np.nonzero(G.v >= 5)[0])

@attr('codegen-independent')
def test_get_dtype():
    '''
    Check the utility function get_dtype
    '''
    eqs = Equations('''dv/dt = -v / (10*ms) : volt
                       x : 1
                       b : boolean
                       n : integer''')

    # Test standard dtypes
    assert get_dtype(eqs['v']) == prefs['core.default_float_dtype']
    assert get_dtype(eqs['x']) == prefs['core.default_float_dtype']
    assert get_dtype(eqs['n']) == prefs['core.default_integer_dtype']
    assert get_dtype(eqs['b']) == np.bool

    # Test a changed default (float) dtype
    assert get_dtype(eqs['v'], np.float32) == np.float32, get_dtype(eqs['v'], np.float32)
    assert get_dtype(eqs['x'], np.float32) == np.float32
    # integer and boolean variables should be unaffected
    assert get_dtype(eqs['n']) == prefs['core.default_integer_dtype']
    assert get_dtype(eqs['b']) == np.bool

    # Explicitly provide a dtype for some variables
    dtypes = {'v': np.float32, 'x': np.float64, 'n': np.int64}
    for varname in dtypes:
        assert get_dtype(eqs[varname], dtypes) == dtypes[varname]

    # Not setting some dtypes should use the standard dtypes
    dtypes = {'n': np.int64}
    assert get_dtype(eqs['n'], dtypes) == np.int64
    assert get_dtype(eqs['v'], dtypes) == prefs['core.default_float_dtype']

    # Test that incorrect types raise an error
    # incorrect general dtype
    assert_raises(TypeError, lambda: get_dtype(eqs['v'], np.int32))
    # incorrect specific types
    assert_raises(TypeError, lambda: get_dtype(eqs['v'], {'v': np.int32}))
    assert_raises(TypeError, lambda: get_dtype(eqs['n'], {'n': np.float32}))
    assert_raises(TypeError, lambda: get_dtype(eqs['b'], {'b': np.int32}))


def test_aliasing_in_statements():
    '''
    Test an issue around variables aliasing other variables (#259)
    '''
    if prefs.codegen.target != 'numpy':
        raise SkipTest('numpy-only test')

    runner_code = '''x_1 = x_0
                     x_0 = -1'''
    g = NeuronGroup(1, model='''x_0 : 1
                                x_1 : 1 ''')
    custom_code_obj = g.custom_operation(runner_code)
    net = Network(g, custom_code_obj)
    net.run(defaultclock.dt)
    assert_equal(g.x_0_[:], np.array([-1]))
    assert_equal(g.x_1_[:], np.array([0]))

@attr('codegen-independent')
def test_get_states():
    G = NeuronGroup(10, '''v : volt
                           x : 1
                           subexpr = x + v/volt : 1
                           subexpr2 = x*volt + v : volt''')
    G.v = 'i*volt'
    G.x = '10*i'
    states_units = G.get_states(['v', 'x', 'subexpr', 'subexpr2'], units=True)
    states = G.get_states(['v', 'x', 'subexpr', 'subexpr2'], units=False)

    assert len(states_units.keys()) == len(states.keys()) == 4
    assert_equal(states_units['v'], np.arange(10)*volt)
    assert_equal(states_units['x'], 10*np.arange(10))
    assert_equal(states_units['subexpr'], 11*np.arange(10))
    assert_equal(states_units['subexpr2'], 11*np.arange(10)*volt)
    assert_equal(states['v'], np.arange(10))
    assert_equal(states['x'], 10*np.arange(10))
    assert_equal(states['subexpr'], 11*np.arange(10))
    assert_equal(states['subexpr2'], 11*np.arange(10))

    all_states = G.get_states(units=True)
    assert set(all_states.keys()) == {'v', 'x', 'subexpr', 'subexpr2', 'N', 't',
                                      'dt', 'i'}


def test_random_vector_values():
    # Make sure that the new "loop-invariant optimisation" does not pull out
    # the random number generation and therefore makes all neurons receiving
    # the same values
    tau = 10*ms
    G = NeuronGroup(100, 'dv/dt = -v / tau + xi*tau**-0.5: 1')
    G.v[:] = 'rand()'
    assert np.var(G.v[:]) > 0
    G.v[:] = 0
    net = Network(G)
    net.run(defaultclock.dt)
    assert np.var(G.v[:]) > 0


if __name__ == '__main__':
    test_creation()
    test_variables()
    test_scalar_variable()
    test_referred_scalar_variable()
    test_linked_variable_correct()
    test_linked_variable_incorrect()
    test_linked_variable_scalar()
    test_linked_variable_indexed()
    test_linked_variable_repeat()
    test_linked_double_linked1()
    test_linked_double_linked2()
    test_linked_double_linked3()
    test_linked_double_linked4()
    test_linked_triple_linked()
    test_linked_subgroup()
    test_linked_subgroup2()
    test_linked_subexpression()
    test_linked_subexpression_2()
    test_linked_subexpression_3()
    test_linked_subexpression_synapse()
    test_linked_variable_indexed_incorrect()
    test_linked_synapses()
    test_linked_var_in_reset()
    test_linked_var_in_reset_size_1()
    test_linked_var_in_reset_incorrect()
    test_stochastic_variable()
    test_stochastic_variable_multiplicative()
    test_unit_errors()
    test_threshold_reset()
    test_unit_errors_threshold_reset()
    test_incomplete_namespace()
    test_namespace_errors()
    test_namespace_warnings()
    test_syntax_errors()
    test_state_variables()
    test_state_variable_access()
    test_state_variable_access_strings()
    test_unknown_state_variables()
    test_subexpression()
    test_subexpression_with_constant()
    test_scalar_parameter_access()
    test_scalar_subexpression()
    test_indices()
    test_repr()
    test_get_dtype()
    if prefs.codegen.target == 'numpy':
        test_aliasing_in_statements()
    test_get_states()
    test_random_vector_values()