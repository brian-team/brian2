from __future__ import division
import uuid

import sympy
import numpy as np
from numpy.testing.utils import assert_raises, assert_equal
from nose import SkipTest, with_setup
from nose.plugins.attrib import attr

from brian2.core.variables import linked_var
from brian2.core.network import Network
from brian2.core.preferences import prefs
from brian2.core.clocks import defaultclock
from brian2.devices.device import reinit_devices, seed
from brian2.equations.equations import Equations
from brian2.groups.group import get_dtype
from brian2.groups.neurongroup import NeuronGroup
from brian2.core.magic import run
from brian2.synapses.synapses import Synapses
from brian2.monitors.statemonitor import StateMonitor
from brian2.units.fundamentalunits import (DimensionMismatchError,
                                           have_same_dimensions)
from brian2.units.unitsafefunctions import linspace
from brian2.units.allunits import second, volt
from brian2.units.stdunits import ms, mV, Hz
from brian2.utils.logger import catch_logs

from brian2.tests.utils import assert_allclose


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
def test_integer_variables_and_mod():
    '''
    Test that integer operations and variable definitions work.
    '''
    n = 10
    eqs = '''
    dv/dt = (a+b+j+k)/second : 1
    j = i%n : integer
    k = i//n : integer
    a = v%(i+1) : 1
    b = v%(2*v) : 1
    '''
    G = NeuronGroup(100, eqs)
    G.v = np.random.rand(len(G))
    run(1*ms)
    assert_equal(G.j[:], G.i[:]%n)
    assert_equal(G.k[:], G.i[:]//n)
    assert_equal(G.a[:], G.v[:]%(G.i[:]+1))

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

@attr('codegen-independent')
def test_variableview_calculations():
    # Check that you can directly calculate with "variable views"
    G = NeuronGroup(10, '''x : 1
                           y : volt''')
    G.x = np.arange(10)
    G.y = np.arange(10)[::-1]*mV
    assert_allclose(G.x * G.y, np.arange(10)*np.arange(10)[::-1]*mV)
    assert_allclose(-G.x, -np.arange(10))
    assert_allclose(-G.y, -np.arange(10)[::-1]*mV)

    assert_allclose(3 * G.x, 3 * np.arange(10))
    assert_allclose(3 * G.y, 3 *np.arange(10)[::-1]*mV)
    assert_allclose(G.x * 3, 3 * np.arange(10))
    assert_allclose(G.y * 3, 3 *np.arange(10)[::-1]*mV)
    assert_allclose(G.x / 2.0, np.arange(10)/2.0)
    assert_allclose(G.y / 2, np.arange(10)[::-1]*mV/2)
    assert_allclose(G.x + 2, 2 + np.arange(10))
    assert_allclose(G.y + 2*mV, 2*mV + np.arange(10)[::-1]*mV)
    assert_allclose(2 + G.x, 2 + np.arange(10))
    assert_allclose(2*mV + G.y, 2*mV + np.arange(10)[::-1]*mV)
    assert_allclose(G.x - 2, np.arange(10) - 2)
    assert_allclose(G.y - 2*mV, np.arange(10)[::-1]*mV - 2*mV)
    assert_allclose(2 - G.x, 2 - np.arange(10))
    assert_allclose(2*mV - G.y, 2*mV - np.arange(10)[::-1]*mV)

    # incorrect units
    assert_raises(DimensionMismatchError, lambda: G.x + G.y)
    assert_raises(DimensionMismatchError, lambda: G.x[:] + G.y)
    assert_raises(DimensionMismatchError, lambda: G.x + G.y[:])
    assert_raises(DimensionMismatchError, lambda: G.x + 3*mV)
    assert_raises(DimensionMismatchError, lambda: 3*mV + G.x)
    assert_raises(DimensionMismatchError, lambda: G.y + 3)
    assert_raises(DimensionMismatchError, lambda: 3 + G.y)


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_stochastic_variable():
    '''
    Test that a NeuronGroup with a stochastic variable can be simulated. Only
    makes sure no error occurs.
    '''
    tau = 10 * ms
    G = NeuronGroup(1, 'dv/dt = -v/tau + xi*tau**-0.5: 1')
    run(defaultclock.dt)

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_stochastic_variable_multiplicative():
    '''
    Test that a NeuronGroup with multiplicative noise can be simulated. Only
    makes sure no error occurs.
    '''
    mu = 0.5/second # drift
    sigma = 0.1/second #diffusion
    G = NeuronGroup(1, 'dX/dt = (mu - 0.5*second*sigma**2)*X + X*sigma*xi*second**.5: 1')
    run(defaultclock.dt)

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
@with_setup(teardown=reinit_devices)
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
    run(.25*second)
    assert_allclose(G2.out[:], np.arange(10)+1)

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_linked_variable_correct():
    '''
    Test correct uses of linked variables.
    '''
    tau = 10*ms
    G1 = NeuronGroup(10, 'dv/dt = -v / tau : volt')
    G1.v = linspace(0*mV, 20*mV, 10)
    G2 = NeuronGroup(10, 'v : volt (linked)')
    G2.v = linked_var(G1.v)
    mon1 = StateMonitor(G1, 'v', record=True)
    mon2 = StateMonitor(G2, 'v', record=True)
    run(10*ms)
    assert_allclose(mon1.v[:, :], mon2.v[:, :])
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
@with_setup(teardown=reinit_devices)
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
    # We don't test anything for now, except that it runs without raising an
    # error
    run(defaultclock.dt)
    # Make sure that printing the variable values works
    assert len(str(G2.x)) > 0
    assert len(repr(G2.x)) > 0
    assert len(str(G2.x[:])) > 0
    assert len(repr(G2.x[:])) > 0
    assert np.isscalar(G2.x[:])
    # Check that subgroups work correctly (see github issue #916)
    sg1 = G2[:5]
    sg2 = G2[5:]
    assert sg1.x == G2.x
    assert sg2.x == G2.x


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
    assert_allclose(G.y[:], np.arange(10)[::-1]*0.1)

@attr('codegen-independent')
def test_linked_variable_repeat():
    '''
    Test a "repeat"-like connection between two groups of different size
    '''
    G1 = NeuronGroup(5, 'w : 1')
    G2 = NeuronGroup(10, 'v : 1 (linked)')
    G2.v = linked_var(G1.w, index=np.arange(5).repeat(2))
    G1.w = np.arange(5) * 0.1
    assert_allclose(G2.v[:], np.arange(5).repeat(2) * 0.1)

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
    assert_allclose(G3.z[:], np.arange(10))

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
    assert_allclose(G3.z[:], np.arange(5).repeat(2)*0.1)

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
    assert_allclose(G3.z[:], np.arange(5).repeat(2)*0.1)

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
    assert_allclose(G3.z[:], np.arange(5).repeat(2)[::-1]*0.1)

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
    assert_allclose(G4.d[:], np.arange(2).repeat(2)[::-1].repeat(2)*0.1)

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

    assert_allclose(G3.y[:], (np.arange(5)+3)*0.1)

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

    assert_allclose(G3.y[:], (np.arange(5)+3).repeat(2)*0.1)

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
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
    run(5*ms)

    # Due to the linking, the first 5 and the second 5 recorded I vectors should
    # be identical
    assert all((all(mon[i].I == mon[0].I) for i in xrange(5)))
    assert all((all(mon[i+5].I == mon[5].I) for i in xrange(5)))

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
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
    run(5*ms)

    assert all(mon[0].I_l == mon1[0].I)
    assert all(mon[1].I_l == mon1[1].I)

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
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
    run(5*ms)

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
    S = Synapses(G, G, 'w:1')
    S.connect()
    G2 = NeuronGroup(100, 'x : 1 (linked)')
    assert_raises(NotImplementedError, lambda: setattr(G2, 'x', linked_var(S, 'w')))

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_linked_var_in_reset():
    G1 = NeuronGroup(3, 'x:1')
    G2 = NeuronGroup(3, '''x_linked : 1 (linked)
                           y:1''',
                     threshold='y>1', reset='y=0; x_linked += 1')
    G2.x_linked = linked_var(G1, 'x')
    G2.y = [0, 1.1, 0]
    # In this context, x_linked should not be considered as a scalar variable
    # and therefore the reset statement should be allowed
    run(3*defaultclock.dt)
    assert_allclose(G1.x[:], [0, 1, 0])

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_linked_var_in_reset_size_1():
    G1 = NeuronGroup(1, 'x:1')
    G2 = NeuronGroup(1, '''x_linked : 1 (linked)
                           y:1''',
                     threshold='y>1', reset='y=0; x_linked += 1')
    G2.x_linked = linked_var(G1, 'x')
    G2.y = 1.1
    # In this context, x_linked should not be considered as a scalar variable
    # and therefore the reset statement should be allowed
    run(3*defaultclock.dt)
    assert_allclose(G1.x[:], 1)

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
    G = NeuronGroup(1, 'dv/dt = -v/tau : 1', threshold='False', reset='v = v_r')
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
        G.x = 'i // N'
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
@with_setup(teardown=reinit_devices)
def test_threshold_reset():
    '''
    Test that threshold and reset work in the expected way.
    '''
    # Membrane potential does not change by itself
    G = NeuronGroup(3, 'dv/dt = 0 / second : 1',
                    threshold='v > 1', reset='v=0.5')
    G.v = np.array([0, 1, 2])
    run(defaultclock.dt)
    assert_allclose(G.v[:], np.array([0, 1, 0.5]))

@attr('codegen-independent')
def test_unit_errors_threshold_reset():
    '''
    Test that unit errors in thresholds and resets are detected.
    '''
    # Unit error in threshold
    group = NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1', threshold='v > -20*mV')
    assert_raises(DimensionMismatchError,
                  lambda: Network(group).run(0*ms))

    # Unit error in reset
    group = NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                        threshold='True',
                        reset='v = -65*mV')
    assert_raises(DimensionMismatchError,
                  lambda: Network(group).run(0*ms))

    # More complicated unit reset with an intermediate variable
    # This should pass
    group = NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                threshold='False',
                reset='''temp_var = -65
                         v = temp_var''')
    run(0*ms)
    # throw in an empty line (should still pass)
    group = NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                threshold='False',
                reset='''temp_var = -65

                         v = temp_var''')
    run(0*ms)
    # This should fail
    group = NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                        threshold='False',
                        reset='''temp_var = -65*mV
                                 v = temp_var''')
    assert_raises(DimensionMismatchError,
                  lambda: Network(group).run(0*ms))

    # Resets with an in-place modification
    # This should work
    group = NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                        threshold='False',
                        reset='''v /= 2''')
    run(0*ms)

    # This should fail
    group = NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                        threshold='False',
                        reset='''v -= 60*mV''')
    assert_raises(DimensionMismatchError,
              lambda: Network(group).run(0*ms))

@attr('codegen-independent')
def test_syntax_errors():
    '''
    Test that syntax errors are already caught at initialization time.
    For equations this is already tested in test_equations
    '''
    
    # We do not specify the exact type of exception here: Python throws a
    # SyntaxError while C++ results in a ValueError
    # Syntax error in threshold
    group = NeuronGroup(1, 'dv/dt = 5*Hz : 1',
                        threshold='>1')
    assert_raises(Exception, lambda: Network(group).run(0*ms))

    # Syntax error in reset
    group = NeuronGroup(1, 'dv/dt = 5*Hz : 1',
                        threshold='True',
                        reset='0')
    assert_raises(Exception, lambda: Network(group).run(0*ms))

@attr('codegen-independent')
def test_custom_events():
    G = NeuronGroup(2, '''event_time1 : second
                          event_time2 : second''',
                    events={'event1': 't>=i*ms and t<i*ms+dt',
                            'event2': 't>=(i+1)*ms and t<(i+1)*ms+dt'})
    G.run_on_event('event1', 'event_time1 = t')
    G.run_on_event('event2', 'event_time2 = t')
    net = Network(G)
    net.run(2.1*ms)
    assert_allclose(G.event_time1[:], [0, 1]*ms)
    assert_allclose(G.event_time2[:], [1, 2]*ms)

def test_custom_events_schedule():
    # In the same time step: event2 will be checked and its code executed
    # before event1 is checked and its code executed
    G = NeuronGroup(2, '''x : 1
                          event_time : second''',
                    events={'event1': 'x>0',
                            'event2': 't>=(i+1)*ms and t<(i+1)*ms+dt'})
    G.set_event_schedule('event1', when='after_resets')
    G.run_on_event('event2', 'x = 1', when='resets')
    G.run_on_event('event1',
                   '''event_time = t
                      x = 0''', when='after_resets', order=1)
    net = Network(G)
    net.run(2.1*ms)
    assert_allclose(G.event_time[:], [1, 2]*ms)


@attr('codegen-independent')
def test_incorrect_custom_event_definition():
    # Incorrect event name
    assert_raises(TypeError, lambda: NeuronGroup(1, '', events={'1event': 'True'}))
    # duplicate definition of 'spike' event
    assert_raises(ValueError, lambda: NeuronGroup(1, '', threshold='True',
                                                  events={'spike': 'False'}))
    # not a threshold
    G = NeuronGroup(1, '', events={'my_event': 10*mV})
    assert_raises(TypeError, lambda: Network(G).run(0*ms))
    # schedule for a non-existing event
    G = NeuronGroup(1, '', threshold='False', events={'my_event': 'True'})
    assert_raises(ValueError, lambda: G.set_event_schedule('another_event'))
    # code for a non-existing event
    assert_raises(ValueError, lambda: G.run_on_event('another_event', ''))


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
    assert_allclose((-G.v)[:], -G.v[:])
    assert_allclose((+G.v)[:], G.v[:])
    #Without units
    assert all(G.v_ - G.v_ == 0)
    assert all(G.v_ - G.v_[:] == 0)
    assert all(G.v_[:] - G.v_ == 0)
    assert all(G.v_ + float(70*mV) == G.v_[:] + float(70*mV))
    assert all(float(70*mV) + G.v_ == G.v_[:] + float(70*mV))
    assert all(G.v_ + G.v_ == 2*G.v_)
    assert all(G.v_ / 2.0 == 0.5*G.v_)
    assert all(1.0 / G.v_ == 1.0 / G.v_[:])
    assert_allclose((-G.v)[:], -G.v[:])
    assert_allclose((+G.v)[:], G.v[:])

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


@attr('codegen-independent')
def test_state_variable_access():
    G = NeuronGroup(10, 'v:volt')
    G.v = np.arange(10) * volt

    assert_allclose(np.asarray(G.v[:]), np.arange(10))
    assert have_same_dimensions(G.v[:], volt)
    assert_allclose(np.asarray(G.v[:]), G.v_[:])
    # Accessing single elements, slices and arrays
    assert G.v[5] == 5 * volt
    assert G.v_[5] == 5
    assert_allclose(G.v[:5], np.arange(5) * volt)
    assert_allclose(G.v_[:5], np.arange(5))
    assert_allclose(G.v[[0, 5]], [0, 5] * volt)
    assert_allclose(G.v_[[0, 5]], np.array([0, 5]))

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
    assert_allclose(G.v['v >= 3*volt'], G.v[3:])
    assert_allclose(G.v_['v >= 3*volt'], G.v_[3:])
    # Should also check for units
    assert_raises(DimensionMismatchError, lambda: G.v['v >= 3'])
    assert_raises(DimensionMismatchError, lambda: G.v['v >= 3*second'])


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_state_variable_set_strings():
    # Instead of overwriting the same variable over and over, we have one
    # variable for each assignment so that we can test everything in the end
    # for standalone.
    G = NeuronGroup(10, '''v1  : volt
                           v2  : volt
                           v3  : volt
                           v4  : volt
                           v5  : volt
                           v6  : volt
                           v7  : volt
                           v7b : volt
                           v7c : volt
                           v8  : volt
                           v9  : volt
                           v9b : volt
                           v9c : volt
                           v10 : volt
                           v11 : volt
                           dv_ref/dt = -v_ref/(10*ms) : 1 (unless refractory)''',
                    threshold='v_ref>1', reset='v_ref=1', refractory=1 * ms)
    # Setting with strings
    # --------------------
    # String value referring to i
    G.v1 = '2*i*volt'
    # String value referring to i
    G.v1[:5] = '3*i*volt'

    # Conditional write variable
    G.v_ref = '2*i'

    G.v2 = np.arange(10)*volt
    # String value referring to a state variable
    G.v2 = '2*v2'
    G.v2[:5] = '2*v2'

    G.v3 = np.arange(10) * volt
    # String value referring to state variables, i, and an external variable
    ext = 5*volt
    G.v3 = 'v3 + ext + (N + i)*volt'

    G.v4 = np.arange(10) * volt
    G.v4[:5] = 'v4 + ext + (N + i)*volt'

    G.v5 = 'v5 + randn()*volt'  # only check that it doesn't raise an error
    G.v5[:5] = 'v5 + randn()*volt'  # only check that it doesn't raise an error

    G.v6 = np.arange(10) * volt
    # String index using a random number
    G.v6['rand() <= 1'] = 0*mV

    G.v7 = np.arange(10) * volt
    # String index referring to i and setting to a scalar value
    G.v7['i>=5'] = 0*mV

    G.v7b = np.arange(10) * volt
    # String index referring to i and setting to a scalar value (no effect)
    G.v7b['i>=10'] = 0*mV

    G.v7c = np.arange(10) * volt
    # String index referring to i and setting to a scalar value (no effect)
    G.v7c['False'] = 0*mV

    G.v8[:5] = np.arange(5) * volt
    # String index referring to a state variable
    G.v8['v8<3*volt'] = 0*mV
    # String index referring to state variables, i, and an external variable
    ext = 2*volt
    G.v8['v8>=ext and i==(N-6)'] = 0*mV

    G.v9 = np.arange(10) * volt
    # Strings for both condition and values
    G.v9['i>=5'] = 'v9*2'
    G.v9['v9<5*volt'] = '3*i*volt'

    G.v9b = np.arange(10) * volt
    # Strings for both condition and values (no effect)
    G.v9b['i<0'] = 'v9 + 100*volt'

    G.v9c = np.arange(10) * volt
    # Strings for both condition and values (no effect)
    G.v9c['False'] = 'v9 + 100*volt'

    G.v10 = np.arange(10)*volt
    G.v10['i<=5'] = '(100 + rand())*volt'

    # string assignment to scalars
    G.v11[0] = '1*volt'
    G.v11[1] = '(1 + i)*volt'
    G.v11[2] = 'v11 + 3*volt'
    G.v11[3] = 'inf*volt'
    G.v11[4] = 'rand()*volt'
    run(0*ms)
    assert_allclose(G.v1[:], [0, 3, 6, 9, 12, 10, 12, 14, 16, 18]*volt)
    assert_allclose(G.v_ref[:], 2 * np.arange(10))
    assert_allclose(G.v2[:], [0, 4, 8, 12, 16, 10, 12, 14, 16, 18]*volt)
    assert_allclose(G.v3[:], 2 * np.arange(10) * volt + 15 * volt)
    assert_allclose(G.v4[:], [15, 17, 19, 21, 23, 5, 6, 7, 8, 9]*volt)
    assert_allclose(G.v6[:], np.zeros(10) * volt)
    assert_allclose(G.v7[:], [0, 1, 2, 3, 4, 0, 0, 0, 0, 0]*volt)
    assert_allclose(G.v7b[:], np.arange(10)*volt)
    assert_allclose(G.v7c[:], np.arange(10) * volt)
    assert_allclose(G.v8[:], [0, 0, 0, 3, 0, 0, 0, 0, 0, 0]*volt)
    assert_allclose(G.v9[:], [0, 3, 6, 9, 12, 10, 12, 14, 16, 18]*volt)
    assert_allclose(G.v9b[:], np.arange(10)*volt)
    assert_allclose(G.v9c[:], np.arange(10) * volt)
    assert_allclose(G.v10[6:], np.arange(4)*volt + 6*volt)  # unchanged
    assert all(G.v10[:6] >= 100*volt)
    assert all(G.v10[:6] <= 101*volt)
    assert np.var(G.v10_[:6]) > 0
    assert_allclose(G.v11[:3], [1, 2, 3]*volt)
    assert np.isinf(G.v11_[3])

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
    assert_allclose(G.expr[:], 2*10*np.arange(10)*Hz + 5*Hz)

@attr('codegen-independent')
def test_subexpression_with_constant():
        g = 2
        G = NeuronGroup(1, '''x : 1
                              I = x*g : 1''')
        G.x = 1
        assert_allclose(G.I[:], np.array([2]))
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
        assert_raises(KeyError, lambda: np.array(G.I))
        assert_raises(KeyError, lambda: np.mean(G.I))
        # But these should
        assert_allclose(np.array(G.I[:]), G.I[:])
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
    assert_allclose(G.freq[:], 100*Hz)
    G.freq[:] = 200*Hz
    assert_allclose(G.freq[:], 200*Hz)
    G.freq = 'freq - 50*Hz + number*Hz'
    assert_allclose(G.freq[:], 150*Hz)
    G.freq[:] = '50*Hz'
    assert_allclose(G.freq[:], 50*Hz)

    # Check the second method of accessing that works
    assert_allclose(np.asanyarray(G.freq), 50*Hz)

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

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_sim_with_scalar_variable():
    G = NeuronGroup(10, '''tau : second (shared)
                           dv/dt = -v/tau : 1''', method='exact')
    G.tau = 10*ms
    G.v = '1.0*i/N'
    run(1*ms)
    assert_allclose(G.v[:], np.exp(-0.1)*np.linspace(0, 1, 10, endpoint=False))


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_sim_with_scalar_subexpression():
    G = NeuronGroup(10, '''tau = 10*ms : second (shared)
                           dv/dt = -v/tau : 1''', method='exact')
    G.v = '1.0*i/N'
    run(1*ms)
    assert_allclose(G.v[:], np.exp(-0.1)*np.linspace(0, 1, 10, endpoint=False))


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_constant_variable_subexpression():
    G = NeuronGroup(10, '''dv1/dt = -v1**2 / (10*ms) : 1
                           dv2/dt = -v_const**2 / (10*ms) : 1
                           dv3/dt = -v_var**2 / (10*ms) : 1
                           dv4/dt = -v_noflag**2 / (10*ms) : 1
                           v_const = v2 : 1 (constant over dt)
                           v_var = v3 : 1
                           v_noflag = v4 : 1''',
                    method='rk2')
    G.v1 = '1.0*i/N'
    G.v2 = '1.0*i/N'
    G.v3 = '1.0*i/N'
    G.v4 = '1.0*i/N'

    run(10*ms)
    # "variable over dt" subexpressions are directly inserted into the equation
    assert_allclose(G.v3[:], G.v1[:])
    assert_allclose(G.v4[:], G.v1[:])
    # "constant over dt" subexpressions will keep a fixed value over the time
    # step and therefore give a slightly different result for multi-step
    # methods
    assert np.sum((G.v2 - G.v1)**2) > 1e-10


@attr('codegen-independent')
def test_constant_subexpression_order():
    G = NeuronGroup(10, '''dv/dt = -v / (10*ms) : 1
                           s1 = v : 1 (constant over dt)
                           s2 = 2*s3 : 1 (constant over dt)
                           s3 = 1 + s1 : 1 (constant over dt)''')
    run(0*ms)
    code_lines = G.subexpression_updater.abstract_code.split('\n')
    assert code_lines[0].startswith('s1')
    assert code_lines[1].startswith('s3')
    assert code_lines[2].startswith('s2')


@attr('codegen-independent')
def test_subexpression_checks():
    group = NeuronGroup(1, '''dv/dt = -v / (10*ms) : volt
                              y = rand() : 1 (constant over dt)
                              z = 17*v**2 : volt**2''')
    # This should all be fine
    net = Network(group)
    net.run(0*ms)

    # The following should raise an error
    group = NeuronGroup(1, '''dv/dt = -v / (10*ms) : volt
                              y = rand() : 1
                              z = 17*v**2 : volt**2''')
    net = Network(group)
    assert_raises(SyntaxError, net.run, 0 * ms)


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
def test_ipython_html():
    G = NeuronGroup(10, '''dv/dt = -(v + Inp) / tau : volt
                           Inp = sin(2*pi*freq*t) : volt
                           freq : Hz''')

    # Test that HTML representation in IPython does not raise errors
    assert len(G._repr_html_())


@attr('codegen-independent')
def test_indices():
    G = NeuronGroup(10, 'v : 1')
    G.v = 'i'
    ext_var = 5
    assert_allclose(G.indices[:], G.i[:])
    assert_allclose(G.indices[5:], G.indices['i >= 5'])
    assert_allclose(G.indices[5:], G.indices['i >= ext_var'])
    assert_allclose(G.indices['v >= 5'], np.nonzero(G.v >= 5)[0])

    # We should not accept "None" as an index, because in numpy this stands for
    # "new axis". In fact, x[0, None] is used in matplotlib to check whether
    # something behaves as a numpy array -- if NeuronGroup accepts None as an
    # index, then synaptic variables will allow indexing in such a way. This
    # makes plotting in matplotlib 1.5.1 fail with a non-obivous error
    # See https://groups.google.com/d/msg/briansupport/yRA4PHKAvN8/cClOEUlOAQAJ
    assert_raises(TypeError, G.indices.__getitem__, None)


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
    g.run_regularly(runner_code)
    net = Network(g)
    net.run(defaultclock.dt)
    assert_allclose(g.x_0_[:], np.array([-1]))
    assert_allclose(g.x_1_[:], np.array([0]))


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
    assert_allclose(states_units['v'], np.arange(10)*volt)
    assert_allclose(states_units['x'], 10*np.arange(10))
    assert_allclose(states_units['subexpr'], 11*np.arange(10))
    assert_allclose(states_units['subexpr2'], 11*np.arange(10)*volt)
    assert_allclose(states['v'], np.arange(10))
    assert_allclose(states['x'], 10*np.arange(10))
    assert_allclose(states['subexpr'], 11*np.arange(10))
    assert_allclose(states['subexpr2'], 11*np.arange(10))

    all_states = G.get_states(units=True)
    assert set(all_states.keys()) == {'v', 'x', 'N', 't', 'dt', 'i'}
    all_states = G.get_states(units=True, subexpressions=True)
    assert set(all_states.keys()) == {'v', 'x', 'N', 't', 'dt', 'i',
                                      'subexpr', 'subexpr2'}


@attr('codegen-independent')
def test_set_states():
    G = NeuronGroup(10, '''v : volt
                           x : 1
                           subexpr = x + v/volt : 1
                           subexpr2 = x*volt + v : volt''')
    G.v = 'i*volt'
    G.x = '10*i'
    assert_raises(ValueError, lambda: G.set_states({'v': np.arange(2, 11)*volt}, units=True))
    # we test if function prevents from setting read_only variables
    assert_raises(TypeError, lambda: G.set_states({'N': 1}))
    assert_raises(DimensionMismatchError, lambda: G.set_states({'x': np.arange(2, 12)*volt}, units=True))
    assert_raises(DimensionMismatchError, lambda: G.set_states({'v': np.arange(2, 12)}, units=True))
    G.set_states({'v': np.arange(2, 12)}, units=False)
    assert_allclose(G.v, np.arange(2, 12)*volt)
    G.set_states({'v': np.arange(2, 12)*volt}, units=True)
    assert_allclose(G.v, np.arange(2, 12)*volt)
    G.set_states({'x': np.arange(2, 12)}, units=False)
    assert_allclose(G.x, np.arange(2, 12))
    G.set_states({'x': np.arange(2, 12)}, units=True)
    assert_allclose(G.x, np.arange(2, 12))


@attr('codegen-independent')
def test_get_states_pandas():
    try:
        import pandas as pd
    except ImportError:
        raise SkipTest('Cannot test export to Pandas data frame, Pandas is not installed.')
    G = NeuronGroup(10, '''v : volt
                           x : 1
                           subexpr = x + v/volt : 1
                           subexpr2 = x*volt + v : volt''')
    G.v = 'i*volt'
    G.x = '10*i'
    assert_raises(NotImplementedError, lambda: G.get_states(['v', 'x', 'subexpr', 'subexpr2'], units=True, format='pandas'))
    states = G.get_states(['v', 'x', 'subexpr', 'subexpr2'], units=False, format='pandas')
    assert_allclose(states['v'].values, np.arange(10))
    assert_allclose(states['x'].values, 10*np.arange(10))
    assert_allclose(states['subexpr'].values, 11*np.arange(10))
    assert_allclose(states['subexpr2'].values, 11*np.arange(10))

    all_states = G.get_states(units=False, format='pandas')
    assert set(all_states.columns) == {'v', 'x', 'N', 't', 'dt', 'i'}
    all_states = G.get_states(units=False, subexpressions=True, format='pandas')
    assert set(all_states.columns) == {'v', 'x', 'N', 't', 'dt', 'i',
                                       'subexpr', 'subexpr2'}


@attr('codegen-independent')
def test_set_states_pandas():
    try:
        import pandas as pd
    except ImportError:
        raise SkipTest('Cannot test export to Pandas data frame, Pandas is not installed.')
    G = NeuronGroup(10, '''v : volt
                           x : 1
                           subexpr = x + v/volt : 1
                           subexpr2 = x*volt + v : volt''')
    G.v = 'i*volt'
    G.x = '10*i'
    df = pd.DataFrame(np.arange(2, 11), columns=['v'])
    assert_raises(NotImplementedError, lambda: G.set_states(df, units=True, format='pandas'))
    assert_raises(ValueError, lambda: G.set_states(df, units=False, format='pandas'))
    # we test if function prevents from setting read_only variables
    df = pd.DataFrame(np.array([1]), columns=['N'])
    assert_raises(TypeError, lambda: G.set_states(df, units=False, format='pandas'))
    df = pd.DataFrame(np.vstack((np.arange(2, 12), np.arange(2, 12))).T)
    df.columns = ['v', 'x']
    G.set_states(df, units=False, format='pandas')
    assert_allclose(G.v, np.arange(2, 12)*volt)
    assert_allclose(G.x, np.arange(2, 12))


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


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_random_values_random_seed():
    G = NeuronGroup(100, '''v1 : 1
                            v2 : 1''')
    seed()
    G.v1 = 'rand() + randn()'
    seed()
    G.v2 = 'rand() + randn()'
    run(0*ms)  # for standalone
    assert np.var(G.v1[:]) > 0
    assert np.var(G.v2[:]) > 0
    assert np.var(G.v1[:] - G.v2[:]) > 0


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_random_values_fixed_seed():
    G = NeuronGroup(100, '''v1 : 1
                            v2 : 1''')
    seed(12345678)
    G.v1 = 'rand() + randn()'
    seed(12345678)
    G.v2 = 'rand() + randn()'
    run(0*ms)  # for standalone
    assert np.var(G.v1[:]) > 0
    assert np.var(G.v2[:]) > 0
    assert_allclose(G.v1[:], G.v2[:])


def test_random_values_fixed_and_random():
    G = NeuronGroup(10, 'dv/dt = -v/(10*ms) + 0.1*xi/sqrt(ms) : 1')
    seed(13579)
    G.v = 'rand()'
    seed()
    mon = StateMonitor(G, 'v', record=True)
    run(2*defaultclock.dt)
    first_run_values = np.array(mon.v)

    G = NeuronGroup(10, 'dv/dt = -v/(10*ms) + 0.1*xi/sqrt(ms) : 1')
    seed(13579)
    G.v = 'rand()'
    seed()
    mon = StateMonitor(G, 'v', record=True)
    run(2*defaultclock.dt)

    # First time step should be identical
    assert_allclose(first_run_values[:, 0], mon.v[:, 0])
    # Second should be different
    assert np.var(first_run_values[:, 1] - mon.v[:, 1]) > 0


@attr('codegen-independent')
def test_no_code():
    # Make sure that we are not unncessarily creating code objects for a state
    # updater that has nothing to do
    group_1 = NeuronGroup(10, 'v: 1', threshold='False')
    # The refractory argument will automatically add a statement for each time
    # step, so we'll need a state updater here
    group_2 = NeuronGroup(10, 'v: 1', threshold='False', refractory=2*ms)
    run(0*ms)
    assert len(group_1.state_updater.code_objects) == 0
    assert group_1.state_updater.codeobj is None
    assert len(group_2.state_updater.code_objects) == 1
    assert group_2.state_updater.codeobj is not None


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_run_regularly_scheduling():
    G = NeuronGroup(1, '''v1 : 1
                          v2 : 1
                          v3 : 1''')
    G.run_regularly('v1 += 1')
    G.run_regularly('v2 = v1', when='end')
    G.run_regularly('v3 = v1', when='before_start')
    run(2*defaultclock.dt)
    assert_allclose(G.v1[:], 2)
    assert_allclose(G.v2[:], 2)
    assert_allclose(G.v3[:], 1)


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_run_regularly_dt():
    G = NeuronGroup(1, 'v : 1')
    G.run_regularly('v += 1', dt=2*defaultclock.dt)
    M = StateMonitor(G, 'v', record=0, when='end')
    run(10 * defaultclock.dt)
    assert_allclose(G.v[:], 5)
    assert_allclose(np.diff(M.v[0]), np.tile([0, 1], 5)[:-1])


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_semantics_floor_division():
    # See github issues #815 and #661
    G = NeuronGroup(11, '''a : integer
                           b : integer
                           x : 1
                           y : 1
                           fvalue : 1
                           ivalue : integer''',
                    dtype={'a': np.int32, 'b': np.int64,
                           'x': np.float, 'y': np.double})
    int_values = np.arange(-5, 6)
    float_values = np.arange(-5.0, 6.0, dtype=np.double)
    G.ivalue = int_values
    G.fvalue = float_values
    with catch_logs() as l:
        G.run_regularly('''
        a = ivalue//3
        b = ivalue//3
        x = fvalue//3
        y = fvalue//3
        ''')
        run(defaultclock.dt)
    assert len(l) == 0
    assert_equal(G.a[:], int_values // 3)
    assert_equal(G.b[:], int_values // 3)
    assert_allclose(G.x[:], float_values // 3)
    assert_allclose(G.y[:], float_values // 3)


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_semantics_floating_point_division():
    # See github issues #815 and #661
    G = NeuronGroup(11, '''x1 : 1
                           x2 : 1
                           y1 : 1
                           y2 : 1
                           fvalue : 1
                           ivalue : integer''',
                    dtype={'a': np.int32, 'b': np.int64,
                           'x': np.float, 'y': np.double})
    int_values = np.arange(-5, 6)
    float_values = np.arange(-5.0, 6.0, dtype=np.double)
    G.ivalue = int_values
    G.fvalue = float_values
    with catch_logs() as l:
        G.run_regularly('''
        x1 = ivalue/3
        x2 = fvalue/3
        y1 = ivalue/3
        y2 = fvalue/3
        ''')
        run(defaultclock.dt)
    # Some devices (e.g. Brian2GeNN) might not raise a warning
    assert (len(l) == 0 or
            (len(l) == 1 and
             l[0][1].endswith('floating_point_division') and
             'ivalue / 3' in l[0][2]))
    assert_allclose(G.x1[:], int_values / 3)
    assert_allclose(G.y1[:], int_values / 3)
    assert_allclose(G.x2[:], float_values / 3)
    assert_allclose(G.y2[:], float_values / 3)


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_semantics_mod():
    # See github issues #815 and #661
    G = NeuronGroup(11, '''a : integer
                           b : integer
                           x : 1
                           y : 1
                           fvalue : 1
                           ivalue : integer''',
                    dtype={'a': np.int32, 'b': np.int64,
                           'x': np.float, 'y': np.double})
    int_values = np.arange(-5, 6)
    float_values = np.arange(-5.0, 6.0, dtype=np.double)
    G.ivalue = int_values
    G.fvalue = float_values
    with catch_logs() as l:
        G.run_regularly('''
        a = ivalue % 3
        b = ivalue % 3
        x = fvalue % 3
        y = fvalue % 3
        ''')
        run(defaultclock.dt)
    assert len(l) == 0
    assert_equal(G.a[:], int_values % 3)
    assert_equal(G.b[:], int_values % 3)
    assert_allclose(G.x[:], float_values % 3)
    assert_allclose(G.y[:], float_values % 3)


if __name__ == '__main__':
    test_set_states()
    test_creation()
    test_integer_variables_and_mod()
    test_variables()
    test_variableview_calculations()
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
    test_threshold_reset()
    test_unit_errors_threshold_reset()
    test_custom_events()
    test_custom_events_schedule()
    test_incorrect_custom_event_definition()
    test_incomplete_namespace()
    test_namespace_errors()
    test_namespace_warnings()
    test_syntax_errors()
    test_state_variables()
    test_state_variable_access()
    test_state_variable_access_strings()
    test_state_variable_set_strings()
    test_unknown_state_variables()
    test_subexpression()
    test_subexpression_with_constant()
    test_scalar_parameter_access()
    test_scalar_subexpression()
    test_sim_with_scalar_variable()
    test_sim_with_scalar_subexpression()
    test_constant_variable_subexpression()
    test_constant_subexpression_order()
    test_subexpression_checks()
    test_indices()
    test_repr()
    test_ipython_html()
    test_get_dtype()
    if prefs.codegen.target == 'numpy':
        test_aliasing_in_statements()
    test_get_states()
    test_set_states()
    test_get_states_pandas()
    test_set_states_pandas()
    test_random_vector_values()
    test_random_values_random_seed()
    test_random_values_fixed_seed()
    test_random_values_fixed_and_random()
    test_no_code()
    test_run_regularly_scheduling()
    test_run_regularly_dt()
    test_semantics_floor_division()
    test_semantics_floating_point_division()
    test_semantics_mod()
