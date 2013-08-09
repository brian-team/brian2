import numpy as np
from numpy.testing.utils import assert_raises, assert_equal, assert_allclose

from brian2.groups.neurongroup import NeuronGroup
from brian2.core.network import Network
from brian2.core.clocks import defaultclock
from brian2.units.fundamentalunits import DimensionMismatchError
from brian2.units.allunits import second
from brian2.units.stdunits import ms, mV
from brian2.codegen.runtime.weave_rt import WeaveCodeObject
from brian2.codegen.runtime.numpy_rt import NumpyCodeObject

# We can only test C++ if weave is availabe
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
    
    G = NeuronGroup(1, 'dv/dt = -v/tau + xi*tau**-0.5: 1')
    assert not 'tau' in G.variables and 'xi' in G.variables


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
    from nose import SkipTest
    raise SkipTest()
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
    from nose import SkipTest
    raise SkipTest()
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
    G = NeuronGroup(10, 'v : volt')
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

    # Calculating with state variables should work too
    assert all(G.v - G.v == 0)

    # And in-place modification should work as well
    G.v += 10*mV
    G.v *= 2
    # with unit checking
    assert_raises(DimensionMismatchError, lambda: G.v.__iadd__(3*second))
    assert_raises(DimensionMismatchError, lambda: G.v.__iadd__(3))
    assert_raises(DimensionMismatchError, lambda: G.v.__imul__(3*second))


if __name__ == '__main__':
    test_creation()
    test_variables()
    test_stochastic_variable()
    test_unit_errors()
    test_threshold_reset()
    test_unit_errors_threshold_reset()
    test_incomplete_namespace()
    test_namespace_errors()
    test_syntax_errors()
    test_state_variables()