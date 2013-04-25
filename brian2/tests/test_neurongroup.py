from numpy.testing.utils import assert_raises

from brian2.groups.neurongroup import NeuronGroup
from brian2.core.network import Network
from brian2.units.fundamentalunits import DimensionMismatchError
from brian2.units.stdunits import ms
from brian2.codegen.languages.cpp import CPPLanguage
from brian2.codegen.languages.python import PythonLanguage

languages = [PythonLanguage(), CPPLanguage()]

def test_creation():
    '''
    A basic test that creating a NeuronGroup works.
    '''
    for language in languages:
        G = NeuronGroup(42, equations='dv/dt = -v/(10*ms) : 1', reset='v=0',
                        threshold='v>1', language=language)
        assert len(G) == 42
    
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

def test_unit_errors_threshold_reset():
    '''
    Test that unit errors in thresholds and resets are detected.
    '''
    from nose import SkipTest
    raise SkipTest('Checking units for threshold/reset not implementd yet.')
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
    test_unit_errors()
    #test_unit_errors_threshold_reset()
    test_incomplete_namespace()
    test_namespace_errors()
    test_syntax_errors()