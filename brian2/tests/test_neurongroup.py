import numpy as np
from numpy.testing.utils import assert_raises, assert_equal

from brian2.groups.neurongroup import NeuronGroup
from brian2.core.network import Network
from brian2.core.clocks import defaultclock
from brian2.units.fundamentalunits import DimensionMismatchError
from brian2.units.stdunits import ms
from brian2.codegen.languages.cpp import CPPLanguage
from brian2.codegen.languages.python import PythonLanguage

# We can only test C++ if weave is availabe
try:
    import scipy.weave
    languages = [PythonLanguage(), CPPLanguage()]
except ImportError:
    # Can't test C++
    languages = [PythonLanguage()]

def test_creation():
    '''
    A basic test that creating a NeuronGroup works.
    '''
    for language in languages:
        G = NeuronGroup(42, equations='dv/dt = -v/(10*ms) : 1', reset='v=0',
                        threshold='v>1', language=language)
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


def test_specifiers():
    '''
    Test the correct creation of the specifiers dictionary.
    '''
    G = NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1')
    assert 'v' in G.specifiers and 't' in G.specifiers and 'dt' in G.specifiers
    
    G = NeuronGroup(1, 'dv/dt = -v/tau + xi*tau**-0.5: 1')
    assert not 'tau' in G.specifiers and 'xi' in G.specifiers


def test_stochastic_variable():
    '''
    Test that a NeuronGroup with a stochastic variable can be simulated. Only
    makes sure no error occurs.
    '''
    tau = 10 * ms
    for language in languages:
        G = NeuronGroup(1, 'dv/dt = -v/tau + xi*tau**-0.5: 1',
                        language=language)
        net = Network(G)
        net.run(defaultclock.dt)

def test_unit_errors():
    '''
    Test that units are checked for a complete namespace.
    '''
    for language in languages:
        # Unit error in model equations
        assert_raises(DimensionMismatchError,
                      lambda: NeuronGroup(1, 'dv/dt = -v : 1',
                                          language=language))
        assert_raises(DimensionMismatchError,
                      lambda: NeuronGroup(1, 'dv/dt = -v/(10*ms) + 2*mV: 1',
                                          language=language))

def test_incomplete_namespace():
    '''
    Test that the namespace does not have to be complete at creation time.
    '''
    for language in languages:
        # This uses tau which is not defined yet (explicit namespace)
        G = NeuronGroup(1, 'dv/dt = -v/tau : 1', namespace={},
                        language=language)
        G.namespace['tau'] = 10*ms
        net = Network(G)
        net.run(1*ms)
        
        # This uses tau which is not defined yet (implicit namespace)
        G = NeuronGroup(1, 'dv/dt = -v/tau : 1',
                        language=language)
        tau = 10*ms
        net = Network(G)
        net.run(1*ms)

def test_namespace_errors():
    
    for language in languages:
        # model equations use unknown identifier
        G = NeuronGroup(1, 'dv/dt = -v/tau : 1', language=language)
        net = Network(G)
        assert_raises(KeyError, lambda: net.run(1*ms))
        
        # reset uses unknown identifier
        G = NeuronGroup(1, 'dv/dt = -v/tau : 1', reset='v = v_r',
                        language=language)
        net = Network(G)
        assert_raises(KeyError, lambda: net.run(1*ms))
        
        # threshold uses unknown identifier
        G = NeuronGroup(1, 'dv/dt = -v/tau : 1', threshold='v > v_th',
                        language=language)
        net = Network(G)
        assert_raises(KeyError, lambda: net.run(1*ms))

def test_threshold_reset():
    '''
    Test that threshold and reset work in the expected way.
    '''
    for language in languages:
        # Membrane potential does not change by itself
        G = NeuronGroup(3, 'dv/dt = 0 / second : 1',
                        threshold='v > 1', reset='v=0.5', language=language)
        G.v = np.array([0, 1, 2])
        net = Network(G)
        net.run(defaultclock.dt)
        assert_equal(G.v, np.array([0, 1, 0.5]))

def test_unit_errors_threshold_reset():
    '''
    Test that unit errors in thresholds and resets are detected.
    '''
    for language in languages:    
        # Unit error in threshold
        assert_raises(DimensionMismatchError,
                      lambda: NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                                          threshold='v > -20*mV',
                                          language=language))
        
        # Unit error in reset
        assert_raises(DimensionMismatchError,
                      lambda: NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                                          reset='v = -65*mV',
                                          language=language))
        
        # More complicated unit reset with an intermediate variable
        # This should pass
        NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                    reset='''temp_var = -65
                             v = temp_var''', language=language)
        # throw in an empty line (should still pass)
        NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                    reset='''temp_var = -65
                    
                             v = temp_var''', language=language)
        
        # This should fail
        assert_raises(DimensionMismatchError,
                      lambda: NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                                          reset='''temp_var = -65*mV
                                                   v = temp_var''',
                                          language=language))
        
        # Resets with an in-place modification
        # This should work
        NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                    reset='''v /= 2''', language=language)
        
        # This should fail
        assert_raises(DimensionMismatchError,
                      lambda: NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1',
                                          reset='''v -= 60*mV''',
                                          language=language))        

def test_syntax_errors():
    '''
    Test that syntax errors are already caught at initialization time.
    For equations this is already tested in test_equations
    '''
    
    # We do not specify the exact type of exception here: Python throws a
    # SyntaxError while C++ results in a ValueError
    
    for language in languages:
    
        # Syntax error in threshold
        assert_raises(Exception,
                      lambda: NeuronGroup(1, 'dv/dt = 5*Hz : 1',
                                          threshold='>1',
                                          language=language),
                      )
    
        # Syntax error in reset
        assert_raises(Exception,
                      lambda: NeuronGroup(1, 'dv/dt = 5*Hz : 1',
                                          reset='0',
                                          language=language))            

if __name__ == '__main__':
    test_creation()
    test_specifiers()
    test_stochastic_variable()
    test_unit_errors()
    test_threshold_reset()
    test_unit_errors_threshold_reset()
    test_incomplete_namespace()
    test_namespace_errors()
    test_syntax_errors()