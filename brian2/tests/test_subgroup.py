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

def test_state_variables_string_indices():
    '''
    Test accessing subgroups with string indices.
    '''
    for codeobj_class in codeobj_classes:
        G = NeuronGroup(10, 'v : volt', codeobj_class=codeobj_class)
        SG = G[4:9]
        assert len(SG.v['i>3']) == 1

        G.v = np.arange(10) * mV
        assert len(SG.v['v>7*mV']) == 1

        # Combined string indexing and assignment
        SG.v['i > 3'] = 'i*10*mV'

        assert_equal(G.v[:], [0, 1, 2, 3, 4, 5, 6, 7, 40, 9] * mV)

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
        G1.v = 'i'
        G2.v = '10 + i'
        SG1 = G1[:5]
        SG2 = G2[10:]
        S = Synapses(SG1, SG2, 'w:1', pre='v+=w', codeobj_class=codeobj_class)
        S.connect(2, 2)  # Should correspond to (2, 12)
        S.connect('i==4 and j==5') # Should correspond to (4, 15)

        # Internally, the "real" neuron indices should be used
        assert_equal(S._synaptic_pre[:], np.array([2, 4]))
        assert_equal(S._synaptic_post[:], np.array([12, 15]))
        # For the user, the subgroup-relative indices should be presented
        assert_equal(S.i[:], np.array([2, 4]))
        assert_equal(S.j[:], np.array([2, 5]))

        # connect based on pre-/postsynaptic state variables
        S = Synapses(SG1, SG2, 'w:1', pre='v+=w', codeobj_class=codeobj_class)
        S.connect('v_pre > 2')
        assert len(S) == 2 * len(SG2), str(len(S))

        S = Synapses(SG1, SG2, 'w:1', pre='v+=w', codeobj_class=codeobj_class)
        S.connect('v_post < 25')
        assert len(S) == 5 * len(SG1), '%s != %s ' % (len(S),5 * len(SG1))


def test_synapse_access():
    for codeobj_class in codeobj_classes:
        G1 = NeuronGroup(10, 'v:1', codeobj_class=codeobj_class)
        G1.v = 'i'
        G2 = NeuronGroup(20, 'v:1', codeobj_class=codeobj_class)
        G2.v = 'i'
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

        # Test referencing pre- and postsynaptic variables
        assert_equal(S.w[2:, :], S.w['v_pre >= 2'])
        assert_equal(S.w[:, :5], S.w['v_post < 15'])
        S.w = 'v_post'
        assert_equal(S.w[:], S.j[:] + 10)
        S.w = 'v_post + v_pre'
        assert_equal(S.w[:], S.j[:] + 10 + S.i[:])


def test_synaptic_propagation():
    for codeobj_class in codeobj_classes:
        G1 = NeuronGroup(10, 'v:1', threshold='v>1', reset='v=0',
                         codeobj_class=codeobj_class)
        G1.v[1::2] = 1.1 # odd numbers should spike
        G2 = NeuronGroup(20, 'v:1', codeobj_class=codeobj_class)
        SG1 = G1[1:6]
        SG2 = G2[10:]
        S = Synapses(SG1, SG2, pre='v+=1', codeobj_class=codeobj_class)
        S.connect('i==j')
        net = Network(G1, G2, S)
        net.run(defaultclock.dt)
        expected = np.zeros(len(G2))
        # Neurons 1, 3, 5 spiked and are connected to 10, 12, 14
        expected[[10, 12, 14]] = 1
        assert_equal(np.asarray(G2.v).flatten(), expected)


def test_spike_monitor():
    for codeobj_class in codeobj_classes:
        G = NeuronGroup(10, 'v:1', threshold='v>1', reset='v=0',
                        codeobj_class=codeobj_class)
        G.v[0] = 1.1
        G.v[2] = 1.1
        G.v[5] = 1.1
        SG = G[3:]
        s_mon = SpikeMonitor(G, codeobj_class=codeobj_class)
        sub_s_mon = SpikeMonitor(SG, codeobj_class=codeobj_class)
        net = Network(G, s_mon, sub_s_mon)
        net.run(defaultclock.dt)
        assert_equal(s_mon.i, np.array([0, 2, 5]))
        assert_equal(s_mon.t_, np.zeros(3))
        assert_equal(sub_s_mon.i, np.array([2]))
        assert_equal(sub_s_mon.t_, np.zeros(1))
        expected = np.zeros(10, dtype=int)
        expected[[0, 2, 5]] = 1
        assert_equal(s_mon.count, expected)
        expected = np.zeros(7, dtype=int)
        expected[[2]] = 1
        assert_equal(sub_s_mon.count, expected)


def test_wrong_indexing():
    G = NeuronGroup(10, 'v:1')
    assert_raises(TypeError, lambda: G[0])
    assert_raises(TypeError, lambda: G[[0, 1]])
    assert_raises(TypeError, lambda: G['string'])

    assert_raises(IndexError, lambda: G[10:])
    assert_raises(IndexError, lambda: G[::2])
    assert_raises(IndexError, lambda: G[3:2])

if __name__ == '__main__':
    test_state_variables()
    test_state_variables_string_indices()
    test_state_monitor()
    test_synapse_creation()
    test_synapse_access()
    test_synaptic_propagation()
    test_spike_monitor()
    test_wrong_indexing()