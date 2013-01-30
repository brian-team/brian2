import numpy as np

from brian2.units import second, volt
from brian2.units.stdunits import ms, Hz, mV
from brian2.units.unitsafefunctions import sin, log, exp
from brian2.core.namespace import Namespace

def test_default_content():
    # any namespace should contain units and some standard functions
    ns = Namespace({}, exhaustive=True)
    assert ns.resolve('ms') is ms
    assert ns.resolve('Hz') is Hz
    assert ns.resolve('mV') is mV
    assert ns.resolve('second') is second
    assert ns.resolve('volt') is volt
    assert ns.resolve('sin') is sin
    assert ns.resolve('log') is log
    assert ns.resolve('exp') is exp
    assert ns.resolve('sqrt') is np.sqrt
    assert ns.resolve('mean') is np.mean


def test_warnings():
    pass

def test_errors():
    pass

def test_resolution_order():
    pass

def test_referred_namespaces():
    pre_ns = {'v': 'v_pre', 'w': 'w_pre'}
    post_ns = {'v': 'v_post', 'w': 'w_post'}
    syn_ns = Namespace({'v': 'v_syn'}, exhaustive=True,
                       refers=[('presynaptic', ['_pre'], pre_ns),
                               ('postsynaptic', ['_post', ''], post_ns)])
    
    # the name in the "Synapse" itself takes precedence
    assert syn_ns['v'] == 'v_syn'
    # Suffixed names should return values from the respective namespaces
    assert syn_ns['v_pre'] == 'v_pre'
    assert syn_ns['v_post'] == 'v_post'
    # An unsuffixed name that does not exist in Synapses itself may refer
    # to the postsynaptic namespace
    assert syn_ns['w'] == 'w_post'
    

if __name__ == '__main__':
    test_default_content()
    test_referred_namespaces()

