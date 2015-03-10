from nose import with_setup
from nose.plugins.attrib import attr
from numpy.testing.utils import assert_raises, assert_equal, assert_allclose

from brian2 import *
from brian2.utils.logger import catch_logs
from brian2.devices.device import restore_device

@attr('codegen-independent')
def test_str_repr():
    '''
    Test the string representation of a subgroup.
    '''
    G = NeuronGroup(10, 'v:1')
    SG = G[5:8]
    # very basic test, only make sure no error is raised
    assert len(str(SG))
    assert len(repr(SG))


def test_state_variables():
    '''
    Test the setting and accessing of state variables in subgroups.
    '''
    G = NeuronGroup(10, 'v : volt')
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

    # Indexing with subgroups
    assert_equal(G.v[SG], SG.v[:])


def test_state_variables_string_indices():
    '''
    Test accessing subgroups with string indices.
    '''
    G = NeuronGroup(10, 'v : volt')
    SG = G[4:9]
    assert len(SG.v['i>3']) == 1

    G.v = np.arange(10) * mV
    assert len(SG.v['v>7*mV']) == 1

    # Combined string indexing and assignment
    SG.v['i > 3'] = 'i*10*mV'

    assert_equal(G.v[:], [0, 1, 2, 3, 4, 5, 6, 7, 40, 9] * mV)

@attr('codegen-independent')
def test_state_variables_group_as_index():
    G = NeuronGroup(10, 'v : 1')
    SG = G[4:9]
    G.v[SG] = 1
    assert_equal(G.v[:], np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0]))
    G.v = 1
    G.v[SG] = '2*v'
    assert_equal(G.v[:], np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 1]))


@attr('codegen-independent')
def test_state_variables_group_as_index_problematic():
    G = NeuronGroup(10, 'v : 1')
    SG = G[4:9]
    G.v = 1
    tests = [('i', 1),
             ('N', 1),
             ('N + i', 2),
             ('v', 0)]
    for value, n_warnings in tests:
        with catch_logs() as l:
            G.v.__setitem__(SG, value)
            assert len(l) == n_warnings, 'expected %d, got %d warnings' % (n_warnings, len(l))
            assert all([entry[1].endswith('ambiguous_string_expression')
                        for entry in l])

@attr('standalone-compatible')
@with_setup(teardown=restore_device)
def test_state_monitor():
    G = NeuronGroup(10, 'v : volt')
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


def test_shared_variable():
    '''Make sure that shared variables work with subgroups'''
    G = NeuronGroup(10, 'v : volt (shared)')
    G.v = 1*volt
    SG = G[5:]
    assert SG.v == 1*volt


def test_synapse_creation():
    G1 = NeuronGroup(10, 'v:1')
    G2 = NeuronGroup(20, 'v:1')
    G1.v = 'i'
    G2.v = '10 + i'
    SG1 = G1[:5]
    SG2 = G2[10:]
    S = Synapses(SG1, SG2, 'w:1', pre='v+=w')
    S.connect(2, 2)  # Should correspond to (2, 12)
    S.connect('i==2 and j==5') # Should correspond to (2, 15)

    # Internally, the "real" neuron indices should be used
    assert_equal(S._synaptic_pre[:], np.array([2, 2]))
    assert_equal(S._synaptic_post[:], np.array([12, 15]))
    # For the user, the subgroup-relative indices should be presented
    assert_equal(S.i[:], np.array([2, 2]))
    assert_equal(S.j[:], np.array([2, 5]))
    # N_incoming and N_outgoing should also be correct
    assert all(S.N_outgoing['i==2'] == 2)
    assert all(S.N_outgoing['i!=2'] == 0)
    assert all(S.N_incoming['j==2 or j==5'] == 1)
    assert all(S.N_incoming['j!=2 and j!=5'] == 0)

    # connect based on pre-/postsynaptic state variables
    S = Synapses(SG1, SG2, 'w:1', pre='v+=w')
    S.connect('v_pre > 2')
    assert len(S) == 2 * len(SG2), str(len(S))

    S = Synapses(SG1, SG2, 'w:1', pre='v+=w')
    S.connect('v_post < 25')
    assert len(S) == 5 * len(SG1), '%s != %s ' % (len(S),5 * len(SG1))


def test_synapse_access():
    G1 = NeuronGroup(10, 'v:1')
    G1.v = 'i'
    G2 = NeuronGroup(20, 'v:1')
    G2.v = 'i'
    SG1 = G1[:5]
    SG2 = G2[10:]
    S = Synapses(SG1, SG2, 'w:1', pre='v+=w')
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

    # Test using subgroups as indices
    assert len(S) == len(S.w[SG1, SG2])
    assert_equal(S.w[SG1, 1], S.w[:, 1])
    assert_equal(S.w[1, SG2], S.w[1, :])
    assert len(S.w[SG1, 10]) == 0


def test_synapses_access_subgroups():
    G1 = NeuronGroup(5, 'x:1')
    G2 = NeuronGroup(10, 'y:1')
    SG1 = G1[2:5]
    SG2 = G2[4:9]
    S = Synapses(G1, G2, 'w:1', connect=True)
    S.w[SG1, SG2] = 1
    assert_equal(S.w['(i>=2 and i<5) and (j>=4 and j<9)'], 1)
    assert_equal(S.w['not ((i>=2 and i<5) and (j>=4 and j<9))'], 0)
    S.w = 0
    S.w[SG1, :] = 1
    assert_equal(S.w['i>=2 and i<5'], 1)
    assert_equal(S.w['not (i>=2 and i<5)'], 0)
    S.w = 0
    S.w[:, SG2] = 1
    assert_equal(S.w['j>=4 and j<9'], 1)
    assert_equal(S.w['not (j>=4 and j<9)'], 0)


@attr('codegen-independent')
def test_synapses_access_subgroups_problematic():
    G1 = NeuronGroup(5, 'x:1')
    G2 = NeuronGroup(10, 'y:1')
    SG1 = G1[2:5]
    SG2 = G2[4:9]
    S = Synapses(G1, G2, 'w:1', connect=True)

    tests = [
        ((SG1, slice(None)), 'i', 1),
        ((SG1, slice(None)), 'i + N_pre', 2),
        ((SG1, slice(None)), 'N_pre', 1),
        ((slice(None), SG2), 'j', 1),
        ((slice(None), SG2), 'N_post', 1),
        ((slice(None), SG2), 'N', 1),
        ((SG1, SG2), 'i', 1),
        ((SG1, SG2), 'i + j', 2),
        ((SG1, SG2), 'N_pre', 1),
        ((SG1, SG2), 'j', 1),
        ((SG1, SG2), 'N_post', 1),
        ((SG1, SG2), 'N', 1),
        # These should not raise a warning
        ((SG1, SG2), 'w', 0),
        ((SG1, SG2), 'x_pre', 0),
        ((SG1, SG2), 'y_post', 0),
        ((SG1, SG2), 'y', 0)
        ]
    for item, value, n_warnings in tests:
        with catch_logs() as l:
            S.w.__setitem__(item, value)
            assert len(l) == n_warnings, 'expected %d, got %d warnings' % (n_warnings, len(l))
            assert all([entry[1].endswith('ambiguous_string_expression')
                        for entry in l])


def test_subexpression_references():
    '''
    Assure that subexpressions in targeted groups are handled correctly.
    '''
    G = NeuronGroup(10, '''v : 1
                           v2 = 2*v : 1''')
    G.v = np.arange(10)
    SG1 = G[:5]
    SG2 = G[5:]

    S1 = Synapses(SG1, SG2, '''w : 1
                          u = v2_post + 1 : 1
                          x = v2_pre + 1 : 1''')
    S1.connect('i==(5-1-j)')
    assert_equal(S1.i[:], np.arange(5))
    assert_equal(S1.j[:], np.arange(5)[::-1])
    assert_equal(S1.u[:], np.arange(10)[:-6:-1]*2+1)
    assert_equal(S1.x[:], np.arange(5)*2+1)

    S2 = Synapses(G, SG2, '''w : 1
                             u = v2_post + 1 : 1
                             x = v2_pre + 1 : 1''')
    S2.connect('i==(5-1-j)')
    assert_equal(S2.i[:], np.arange(5))
    assert_equal(S2.j[:], np.arange(5)[::-1])
    assert_equal(S2.u[:], np.arange(10)[:-6:-1]*2+1)
    assert_equal(S2.x[:], np.arange(5)*2+1)

    S3 = Synapses(SG1, G, '''w : 1
                             u = v2_post + 1 : 1
                             x = v2_pre + 1 : 1''')
    S3.connect('i==(10-1-j)')
    assert_equal(S3.i[:], np.arange(5))
    assert_equal(S3.j[:], np.arange(10)[:-6:-1])
    assert_equal(S3.u[:], np.arange(10)[:-6:-1]*2+1)
    assert_equal(S3.x[:], np.arange(5)*2+1)


def test_subexpression_no_references():
    '''
    Assure that subexpressions  are handled correctly, even
    when the subgroups are created on-the-fly.
    '''
    G = NeuronGroup(10, '''v : 1
                           v2 = 2*v : 1''')
    G.v = np.arange(10)

    assert_equal(G[5:].v2, np.arange(5, 10)*2)

    S1 = Synapses(G[:5], G[5:], '''w : 1
                          u = v2_post + 1 : 1
                          x = v2_pre + 1 : 1''')
    S1.connect('i==(5-1-j)')
    assert_equal(S1.i[:], np.arange(5))
    assert_equal(S1.j[:], np.arange(5)[::-1])
    assert_equal(S1.u[:], np.arange(10)[:-6:-1]*2+1)
    assert_equal(S1.x[:], np.arange(5)*2+1)

    S2 = Synapses(G, G[5:], '''w : 1
                             u = v2_post + 1 : 1
                             x = v2_pre + 1 : 1''')
    S2.connect('i==(5-1-j)')
    assert_equal(S2.i[:], np.arange(5))
    assert_equal(S2.j[:], np.arange(5)[::-1])
    assert_equal(S2.u[:], np.arange(10)[:-6:-1]*2+1)
    assert_equal(S2.x[:], np.arange(5)*2+1)

    S3 = Synapses(G[:5], G, '''w : 1
                             u = v2_post + 1 : 1
                             x = v2_pre + 1 : 1''')
    S3.connect('i==(10-1-j)')
    assert_equal(S3.i[:], np.arange(5))
    assert_equal(S3.j[:], np.arange(10)[:-6:-1])
    assert_equal(S3.u[:], np.arange(10)[:-6:-1]*2+1)
    assert_equal(S3.x[:], np.arange(5)*2+1)

@attr('standalone-compatible')
@with_setup(teardown=restore_device)
def test_synaptic_propagation():
    G1 = NeuronGroup(10, 'v:1', threshold='v>1', reset='v=0')
    G1.v['i%2==1'] = 1.1 # odd numbers should spike
    G2 = NeuronGroup(20, 'v:1')
    SG1 = G1[1:6]
    SG2 = G2[10:]
    S = Synapses(SG1, SG2, pre='v+=1')
    S.connect('i==j')
    net = Network(G1, G2, S)
    net.run(defaultclock.dt)
    expected = np.zeros(len(G2))
    # Neurons 1, 3, 5 spiked and are connected to 10, 12, 14
    expected[[10, 12, 14]] = 1
    assert_equal(np.asarray(G2.v).flatten(), expected)

@attr('standalone-compatible')
@with_setup(teardown=restore_device)
def test_spike_monitor():
    G = NeuronGroup(10, 'v:1', threshold='v>1', reset='v=0')
    G.v[0] = 1.1
    G.v[2] = 1.1
    G.v[5] = 1.1
    SG = G[3:]
    s_mon = SpikeMonitor(G)
    sub_s_mon = SpikeMonitor(SG)
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


@attr('codegen-independent')
def test_wrong_indexing():
    G = NeuronGroup(10, 'v:1')
    assert_raises(TypeError, lambda: G[0])
    assert_raises(TypeError, lambda: G[[0, 1]])
    assert_raises(TypeError, lambda: G['string'])

    assert_raises(IndexError, lambda: G[10:])
    assert_raises(IndexError, lambda: G[::2])
    assert_raises(IndexError, lambda: G[3:2])

def test_no_reference_1():
    '''
    Using subgroups without keeping an explicit reference. Basic access.
    '''
    G = NeuronGroup(10, 'v:1')
    G.v = np.arange(10)
    assert_equal(G[:5].v[:], G.v[:5])

@attr('standalone-compatible')
@with_setup(teardown=restore_device)
def test_no_reference_2():
    '''
    Using subgroups without keeping an explicit reference. Monitors
    '''
    G = NeuronGroup(2, 'v:1', threshold='v>1', reset='v=0')
    G.v = [0, 1.1]
    state_mon = StateMonitor(G[:1], 'v', record=True)
    spike_mon = SpikeMonitor(G[1:])
    rate_mon = PopulationRateMonitor(G[:2])
    net = Network(G, state_mon, spike_mon, rate_mon)
    net.run(2*defaultclock.dt)
    assert_equal(state_mon[0].v[:], np.zeros(2))
    assert_equal(spike_mon.i[:], np.array([0]))
    assert_equal(spike_mon.t[:], np.array([0])*second)
    assert_equal(rate_mon.rate[:], np.array([0.5, 0])/defaultclock.dt)

@attr('standalone-compatible')
@with_setup(teardown=restore_device)
def test_no_reference_3():
    '''
    Using subgroups without keeping an explicit reference. Monitors
    '''
    G = NeuronGroup(2, 'v:1', threshold='v>1', reset='v=0')
    G.v = [1.1, 0]
    S = Synapses(G[:1], G[1:], pre='v+=1', connect=True)
    net = Network(G, S)
    net.run(defaultclock.dt)
    assert_equal(G.v[:], np.array([0, 1]))

@attr('standalone-compatible')
@with_setup(teardown=restore_device)
def test_no_reference_4():
    '''
    Using subgroups without keeping an explicit reference. Synapses
    '''
    G1 = NeuronGroup(10, 'v:1', threshold='v>1', reset='v=0')
    G1.v['i%2==1'] = 1.1 # odd numbers should spike
    G2 = NeuronGroup(20, 'v:1')
    S = Synapses(G1[1:6], G2[10:], pre='v+=1')
    S.connect('i==j')
    net = Network(G1, G2, S)
    net.run(defaultclock.dt)
    expected = np.zeros(len(G2))
    # Neurons 1, 3, 5 spiked and are connected to 10, 12, 14
    expected[[10, 12, 14]] = 1
    assert_equal(np.asarray(G2.v).flatten(), expected)


def test_recursive_subgroup():
    '''
    Create a subgroup of a subgroup
    '''
    G = NeuronGroup(10, 'v : 1')
    G.v = 'i'
    SG = G[3:8]
    SG2 = SG[2:4]
    assert_equal(SG2.v[:], np.array([5, 6]))
    assert_equal(SG2.v[:], SG.v[2:4])
    assert SG2.source.name == G.name

if __name__ == '__main__':
    test_str_repr()
    test_state_variables()
    test_state_variables_string_indices()
    test_state_variables_group_as_index()
    test_state_variables_group_as_index_problematic()
    test_state_monitor()
    test_shared_variable()
    test_synapse_creation()
    test_synapse_access()
    test_synapses_access_subgroups()
    test_synapses_access_subgroups_problematic()
    test_subexpression_references()
    test_subexpression_no_references()
    test_synaptic_propagation()
    test_spike_monitor()
    test_wrong_indexing()
    test_no_reference_1()
    test_no_reference_2()
    test_no_reference_3()
    test_no_reference_4()
    test_recursive_subgroup()
