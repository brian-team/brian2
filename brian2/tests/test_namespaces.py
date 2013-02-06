from numpy.testing.utils import assert_raises

from brian2.units import second, volt
from brian2.units.stdunits import ms, Hz, mV
from brian2.units.unitsafefunctions import sin, log, exp
from brian2.core.namespace import (ObjectWithNamespace,
                                   ModelNamespace, Namespace)
from brian2.utils.logger import catch_logs

def _assert_one_warning(l):
    assert len(l) == 1, "expected one warning got %d" % len(l)
    assert l[0][0] == 'WARNING', "expected a WARNING, got %s instead" % l[0][0]

def test_default_content():
    '''
    Test that the default namespace contains standard units and functions.
    '''
    obj = ObjectWithNamespace()
    namespace = obj.create_namespace({})
    # Units
    assert namespace['second'] == second
    assert namespace['volt'] == volt
    assert namespace['ms'] == ms
    assert namespace['Hz'] == Hz
    assert namespace['mV'] == mV
    # Functions
    assert namespace['sin'] == sin
    assert namespace['log'] == log
    assert namespace['exp'] == exp

def test_explicit_namespace():
    ''' Test resolution with an explicitly provided namespace '''
    
    obj = ObjectWithNamespace()
    model_namespace = {'variable': 'model_var'}
    explicit_namespace = {'variable': 'explicit_var',
                          'sin': 'explicit_sin'}
    # Explicitly provided 
    namespace = obj.create_namespace(model_namespace,
                                     explicit_namespace)
    
    
    with catch_logs() as l:
        # model variables takes precedence
        assert namespace['variable'] == 'model_var'
        _assert_one_warning(l)
    
    with catch_logs() as l:
        # The explicitly provided namespace should take precedence over
        # the standard function namespace
        assert namespace['sin'] == 'explicit_sin'
        _assert_one_warning(l)


def test_implicit_namespace():
    ''' Test resolution with an implicitly provided namespace '''
    
    # import something into the local namespace
    from brian2.units.unitsafefunctions import sin
    
    variable = 'local_variable'
    variable2 = 'local_variable2'
    
    obj = ObjectWithNamespace()
    model_namespace = {'variable': 'model_var'}
    
    # No explicitly provided namespace --> use locals and globals 
    namespace = obj.create_namespace(model_namespace)
    
    with catch_logs() as l:
        # model variables take precedence
        assert namespace['variable'] == 'model_var'
        _assert_one_warning(l)
    
    with catch_logs() as l:
        # no conflict here
        assert namespace['variable2'] == 'local_variable2'
        assert len(l) == 0
    
    with catch_logs() as l:
        assert namespace['sin'] == sin
        # There is a conflict here: sin is in the local namespace but also in
        # the default numpy namespace. We do *not* want to raise a warning here
        # however as both refer to the same thing
        assert len(l) == 0  

def test_errors():
    obj = ObjectWithNamespace()
    model_namespace = {'variable': 'model_var'}
    
    namespace = obj.create_namespace(model_namespace)
    
    assert_raises(KeyError, lambda: namespace['nonexisting_variable'])


def test_referred_namespaces():
    pre_ns = ModelNamespace({'v': 'v_pre', 'w': 'w_pre'})
    post_ns = ModelNamespace({'v': 'v_post', 'w': 'w_post'})
    syn_ns = ModelNamespace({'v': 'v_syn'})
    syn_ns.add_namespace(Namespace('presynaptic',
                                   pre_ns,
                                   suffixes=['_pre']))
    syn_ns.add_namespace(Namespace('postsynaptic',
                                   post_ns,
                                   suffixes=['_post', '']))
    
    # the name in the "Synapse" itself takes precedence (but should raise a
    # warning)
    with catch_logs() as l:
        assert syn_ns['v'] == 'v_syn'
        _assert_one_warning(l)
        
    # Suffixed names should return values from the respective namespaces
    assert syn_ns['v_pre'] == 'v_pre'
    assert syn_ns['v_post'] == 'v_post'
    # An unsuffixed name that does not exist in Synapses itself may refer
    # to the postsynaptic namespace
    assert syn_ns['w'] == 'w_post'
    

if __name__ == '__main__':
    test_default_content()
    test_explicit_namespace()
    test_implicit_namespace()
    test_errors()
    test_referred_namespaces()
