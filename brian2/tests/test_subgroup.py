import numpy as np
from numpy.testing.utils import assert_raises, assert_equal, assert_allclose

from brian2 import *

# We can only test C++ if weave is availabe
try:
    import scipy.weave
    codeobj_classes = [NumpyCodeObject, WeaveCodeObject]
except ImportError:
    # Can't test C++
    codeobj_classes = [NumpyCodeObject]



def test_state_variables():
    '''
    Test the setting and accessing of state variables in subgroups.
    '''
    for codeobj_class in codeobj_classes:
        G = NeuronGroup(10, 'v : volt', codeobj_class=codeobj_class)
        SG = G[4:9]
        assert_raises(DimensionMismatchError, lambda: SG.__setattr__('v', -70))
        SG.v_ = float(-80*mV)
        assert_allclose(G.v,
                        np.array([0, 0, 0, 0, -80, -80, -80, -80, -80, 0])*mV)
        assert_allclose(SG.v,
                        np.array([-80, -80, -80, -80, -80])*mV)
        assert_allclose(G.v_,
                        np.array([0, 0, 0, 0, -80, -80, -80, -80, -80, 0])*mV)
        assert_allclose(SG.v_,
                        np.array([-80, -80, -80, -80, -80])*mV)
        # You should also be able to set variables with a string
        SG.v = 'v + i*mV'
        assert_allclose(SG.v[0], -80*mV)
        assert_allclose(SG.v[4], -76*mV)
        assert_allclose(G.v[4:9], -80*mV + np.arange(5)*mV)

        # Calculating with state variables should work too
        assert all(G.v[4:9] - SG.v == 0)

        # And in-place modification should work as well
        SG.v += 10*mV
        assert_allclose(G.v[4:9], -70*mV + np.arange(5)*mV)
        SG.v *= 2
        assert_allclose(G.v[4:9], 2*(-70*mV + np.arange(5)*mV))
        # with unit checking
        assert_raises(DimensionMismatchError, lambda: SG.v.__iadd__(3*second))
        assert_raises(DimensionMismatchError, lambda: SG.v.__iadd__(3))
        assert_raises(DimensionMismatchError, lambda: SG.v.__imul__(3*second))


def test_state_monitor():
    for codeobj_class in codeobj_classes:
        G = NeuronGroup(10, 'v : volt', codeobj_class=codeobj_class)
        G.v = np.arange(10) * volt
        SG = G[5:]
        mon_all = StateMonitor(SG, 'v', record=True)
        mon_0 = StateMonitor(SG, 'v', record=0)
        net = Network(G, SG, mon_all, mon_0)
        net.run(defaultclock.dt)

        assert_equal(mon_0[0].v, mon_all[0].v)
        assert_equal(mon_0[0].v, np.array([5]) * volt)
        assert_equal(mon_all.v.flatten(), np.arange(5, 10) * volt)

        assert_raises(IndexError, lambda: mon_all[5])


def test_synapse_creation():
    for codeobj_class in codeobj_classes:
        G1 = NeuronGroup(10, 'v:1', codeobj_class=codeobj_class)
        G2 = NeuronGroup(20, 'v:1', codeobj_class=codeobj_class)
        SG1 = G1[:5]
        SG2 = G2[10:]
        S = Synapses(SG1, SG2, 'w:1', pre='v+=w', codeobj_class=codeobj_class)
        S.connect(2, 2)  # Should correspond to (2, 12)
        S.connect('i==4 and j==5') # Should correspond to (4, 15)

        # The "true" indices
        assert_equal(S.item_mapping.synaptic_pre, np.array([2, 4]))
        assert_equal(S.item_mapping.synaptic_post, np.array([12, 15]))

        # S.i and S.j should give "subgroup-relative" numbers
        assert_equal(S.i[:], np.array([2, 4]))
        assert_equal(S.j[:], np.array([2, 5]))


def test_synapse_access():
    for codeobj_class in codeobj_classes:
        G1 = NeuronGroup(10, 'v:1', codeobj_class=codeobj_class)
        G2 = NeuronGroup(20, 'v:1', codeobj_class=codeobj_class)
        SG1 = G1[:5]
        SG2 = G2[10:]
        S = Synapses(SG1, SG2, 'w:1', pre='v+=w', codeobj_class=codeobj_class)
        S.connect(True)
        S.w['j == 0'] = 5
        assert all(S.w['j==0'] == 5)
        S.w[2, 2] = 7
        assert all(S.w['i==2 and j==2'] == 7)
        S.w = '2*j'
        assert all(S.w[:, 1] == 2)

        assert len(S.w[:, 10]) == 0
        assert len(S.w['j==10']) == 0

if __name__ == '__main__':
    test_state_variables()
    test_state_monitor()
    test_synapse_creation()
    test_synapse_access()
