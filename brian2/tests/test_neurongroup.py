import sympy
import numpy as np
from numpy.testing.utils import assert_raises, assert_equal, assert_allclose

from brian2.groups.neurongroup import NeuronGroup
from brian2.core.network import Network
from brian2.core.clocks import defaultclock
from brian2.units.fundamentalunits import (DimensionMismatchError,
                                           have_same_dimensions)
from brian2.units.allunits import second, volt
from brian2.units.stdunits import ms, mV
from brian2.codegen.runtime.weave_rt import WeaveCodeObject
from brian2.codegen.runtime.numpy_rt import NumpyCodeObject
from brian2.utils.logger import catch_logs

# We can only test C++ if weave is available
try:
    import scipy.weave
    codeobj_classes = [NumpyCodeObject, WeaveCodeObject]
except ImportError:
    # Can't test C++
    codeobj_classes = [NumpyCodeObject]


def test_creation():
    '''
    A basic test that creating a NeuronGroup works.
    '''
    for codeobj_class in codeobj_classes:
        G = NeuronGroup(42, model='dv/dt = -v/(10*ms) : 1', reset='v=0',
                        threshold='v>1', codeobj_class=codeobj_class)
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


def test_stochastic_variable():
    '''
    Test that a NeuronGroup with a stochastic variable can be simulated. Only
    makes sure no error occurs.
    '''
    tau = 10 * ms
    for codeobj_class in codeobj_classes:
        G = NeuronGroup(1, 'dv/dt = -v/tau + xi*tau**-0.5: 1',
                        codeobj_class=codeobj_class)
        net = Network(G)
        net.run(defaultclock.dt)

def test_unit_errors():
    '''
    Test that units are checked for a complete namespace.
    '''
    for codeobj_class in codeobj_classes:
        # Unit error in model equations
        assert_raises(DimensionMismatchError,
                      lambda: NeuronGroup(1, 'dv/dt = -v : 1',
                                          codeobj_class=codeobj_class))
        assert_raises(DimensionMismatchError,
                      lambda: NeuronGroup(1, 'dv/dt = -v/(10*ms) + 2*mV: 1',
                                          codeobj_class=codeobj_class))

def test_incomplete_namespace():
    '''
    Test that the namespace does not have to be complete at creation time.
    '''
    for codeobj_class in codeobj_classes:
        # This uses tau which is not defined yet (explicit namespace)
        G = NeuronGroup(1, 'dv/dt = -v/tau : 1', namespace={},
                        codeobj_class=codeobj_class)
        G.namespace['tau'] = 10*ms
        net = Network(G)
        net.run(1*ms)
        
        # This uses tau which is not defined yet (implicit namespace)
        G = NeuronGroup(1, 'dv/dt = -v/tau : 1',
                        codeobj_class=codeobj_class)
        tau = 10*ms
        net = Network(G)
        net.run(1*ms)

def test_namespace_errors():
    
    for codeobj_class in codeobj_classes:
        # model equations use unknown identifier
        G = NeuronGroup(1, 'dv/dt = -v/tau : 1', codeobj_class=codeobj_class)
        net = Network(G)
        assert_raises(KeyError, lambda: net.run(1*ms))
        
        # reset uses unknown identifier
        G = NeuronGroup(1, 'dv/dt = -v/tau : 1', reset='v = v_r',
                        codeobj_class=codeobj_class)
        net = Network(G)
        assert_raises(KeyError, lambda: net.run(1*ms))
        
        # threshold uses unknown identifier
        G = NeuronGroup(1, 'dv/dt = -v/tau : 1', threshold='v > v_th',
                        codeobj_class=codeobj_class)
        net = Network(G)
        assert_raises(KeyError, lambda: net.run(1*ms))


def test_namespace_warnings():
    G = NeuronGroup(1, '''x : 1
                          y : 1''')
    # conflicting variable in namespace
    y = 5
    with catch_logs() as l:
        G.x = 'y'
        assert len(l) == 1
        assert l[0][1].endswith('.resolution_conflict')

    # conflicting variables with special meaning
    i = 5
    N = 3
    with catch_logs() as l:
        G.x = 'i / N'
        assert len(l) == 2
        assert l[0][1].endswith('.resolution_conflict')
        assert l[1][1].endswith('.resolution_conflict')


def test_threshold_reset():
    '''
    Test that threshold and reset work in the expected way.
    '''
    for codeobj_class in codeobj_classes:
        # Membrane potential does not change by itself
        G = NeuronGroup(3, 'dv/dt = 0 / second : 1',
                        threshold='v > 1', reset='v=0.5', codeobj_class=codeobj_class)
        G.v = np.array([0, 1, 2])
        net = Network(G)
        net.run(defaultclock.dt)
        assert_equal(G.v[:], np.array([0, 1, 0.5]))

def test_unit_errors_threshold_reset():
    '''
    Test that unit errors in thresholds and resets are detected.
    '''
    for codeobj_class in codeobj_classes:    
        # Unit error in threshold
        assert_raises(DimensionMismatchError,
                      lambda: NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                                          threshold='v > -20*mV',
                                          codeobj_class=codeobj_class))
        
        # Unit error in reset
        assert_raises(DimensionMismatchError,
                      lambda: NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                                          reset='v = -65*mV',
                                          codeobj_class=codeobj_class))
        
        # More complicated unit reset with an intermediate variable
        # This should pass
        NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                    reset='''temp_var = -65
                             v = temp_var''', codeobj_class=codeobj_class)
        # throw in an empty line (should still pass)
        NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                    reset='''temp_var = -65
                    
                             v = temp_var''', codeobj_class=codeobj_class)
        
        # This should fail
        assert_raises(DimensionMismatchError,
                      lambda: NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                                          reset='''temp_var = -65*mV
                                                   v = temp_var''',
                                          codeobj_class=codeobj_class))
        
        # Resets with an in-place modification
        # This should work
        NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                    reset='''v /= 2''', codeobj_class=codeobj_class)
        
        # This should fail
        assert_raises(DimensionMismatchError,
                      lambda: NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                                          reset='''v -= 60*mV''',
                                          codeobj_class=codeobj_class))        

def test_syntax_errors():
    '''
    Test that syntax errors are already caught at initialization time.
    For equations this is already tested in test_equations
    '''
    
    # We do not specify the exact type of exception here: Python throws a
    # SyntaxError while C++ results in a ValueError
    for codeobj_class in codeobj_classes:
    
        # Syntax error in threshold
        assert_raises(Exception,
                      lambda: NeuronGroup(1, 'dv/dt = 5*Hz : 1',
                                          threshold='>1',
                                          codeobj_class=codeobj_class),
                      )
    
        # Syntax error in reset
        assert_raises(Exception,
                      lambda: NeuronGroup(1, 'dv/dt = 5*Hz : 1',
                                          reset='0',
                                          codeobj_class=codeobj_class))            


def test_state_variables():
    '''
    Test the setting and accessing of state variables.
    '''
    for codeobj_class in codeobj_classes:
        G = NeuronGroup(10, 'v : volt', codeobj_class=codeobj_class)

        # The variable N should be always present
        assert G.N == 10
        # But it should be read-only
        assert_raises(TypeError, lambda: G.__setattr__('N', 20))
        assert_raises(TypeError, lambda: G.__setattr__('N_', 20))

        G.v = -70*mV
        assert_raises(DimensionMismatchError, lambda: G.__setattr__('v', -70))
        G.v_ = float(-70*mV)
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
    for codeobj_class in codeobj_classes:
        G = NeuronGroup(10, 'v:volt', codeobj_class=codeobj_class)
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
    for codeobj_class in codeobj_classes:
        G = NeuronGroup(10, 'v:volt', codeobj_class=codeobj_class)
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


if __name__ == '__main__':
    test_creation()
    test_variables()
    test_stochastic_variable()
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
    test_repr()
