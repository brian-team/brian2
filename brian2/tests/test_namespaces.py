import numpy as np

from brian2.units import second, volt
from brian2.units.stdunits import ms, Hz, mV
from brian2.units.unitsafefunctions import sin, log, exp
from brian2.core.namespace import ModelNamespace, Namespace
from brian2.utils.logger import catch_logs

def test_default_content():
    pass

def test_warnings():
    pass

def test_errors():
    pass

def test_resolution_order():
    pass

def test_referred_namespaces():
    pre_ns = ModelNamespace({'v': 'v_pre', 'w': 'w_pre'})
    post_ns = ModelNamespace({'v': 'v_post', 'w': 'w_post'})
    syn_ns = ModelNamespace({'v': 'v_syn'})
    syn_ns.add_namespace(Namespace('presynaptic',
                                   pre_ns.namespaces['model'],
                                   suffixes=['_pre']))
    syn_ns.add_namespace(Namespace('postsynaptic',
                                   post_ns.namespaces['model'],
                                   suffixes=['_post', '']))
    
    # the name in the "Synapse" itself takes precedence (but should raise a
    # warning)
    with catch_logs() as l:
        assert syn_ns['v'] == 'v_syn'
        assert len(l) == 1  # one warning
        assert l[0][0] == 'WARNING'
        
    # Suffixed names should return values from the respective namespaces
    assert syn_ns['v_pre'] == 'v_pre'
    assert syn_ns['v_post'] == 'v_post'
    # An unsuffixed name that does not exist in Synapses itself may refer
    # to the postsynaptic namespace
    assert syn_ns['w'] == 'w_post'
    

if __name__ == '__main__':
    test_default_content()
    test_referred_namespaces()

