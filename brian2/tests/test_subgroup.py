from nose import with_setup
from nose.plugins.attrib import attr
from numpy.testing.utils import assert_raises, assert_equal, assert_allclose

from brian2 import *
from brian2.utils.logger import catch_logs
from brian2.devices.device import reinit_devices

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
                    np.array([0, 0, 0, 0, -80, -80, -80, -80, -80, 0]*mV))
    assert_allclose(SG.v_,
                    np.array([-80, -80, -80, -80, -80]*mV))
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

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_state_variables_simple():
    G = NeuronGroup(10, '''a : 1
                           b : 1
                           c : 1
                           d : 1
                           ''')
    SG = G[3:7]
    SG.a = 1
    SG.a['i == 0'] = 2
    SG.b = 'i'
    SG.b['i == 3'] = 'i * 2'
    SG.c = np.arange(3, 7)
    SG.d[1:2] = 4
    SG.d[2:4] = [1, 2]
    run(0*ms)
    assert_equal(G.a[:], [0, 0, 0, 2, 1, 1, 1, 0, 0, 0])
    assert_equal(G.b[:], [0, 0, 0, 0, 1, 2, 6, 0, 0, 0])
    assert_equal(G.c[:], [0, 0, 0, 3, 4, 5, 6, 0, 0, 0])
    assert_equal(G.d[:], [0, 0, 0, 0, 4, 1, 2, 0, 0, 0])


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
@with_setup(teardown=reinit_devices)
def test_state_monitor():
    G = NeuronGroup(10, 'v : volt')
    G.v = np.arange(10) * volt
    SG = G[5:]
    mon_all = StateMonitor(SG, 'v', record=True)
    mon_0 = StateMonitor(SG, 'v', record=0)
    run(defaultclock.dt)

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


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_synapse_creation():
    G1 = NeuronGroup(10, 'v:1', threshold='False')
    G2 = NeuronGroup(20, 'v:1', threshold='False')
    G1.v = 'i'
    G2.v = '10 + i'
    SG1 = G1[:5]
    SG2 = G2[10:]
    S = Synapses(SG1, SG2, 'w:1', on_pre='v+=w')
    S.connect(i=2, j=2)  # Should correspond to (2, 12)
    S.connect('i==2 and j==5') # Should correspond to (2, 15)

    # connect based on pre-/postsynaptic state variables
    S2 = Synapses(SG1, SG2, 'w:1', on_pre='v+=w')
    S2.connect('v_pre > 2')

    S3 = Synapses(SG1, SG2, 'w:1', on_pre='v+=w')
    S3.connect('v_post < 25')

    S4 = Synapses(SG2, SG1, 'w:1', on_pre='v+=w')
    S4.connect('v_post > 2')

    S5 = Synapses(SG2, SG1, 'w:1', on_pre='v+=w')
    S5.connect('v_pre < 25')

    run(0*ms)  # for standalone

    # Internally, the "real" neuron indices should be used
    assert_equal(S._synaptic_pre[:], np.array([2, 2]))
    assert_equal(S._synaptic_post[:], np.array([12, 15]))
    # For the user, the subgroup-relative indices should be presented
    assert_equal(S.i[:], np.array([2, 2]))
    assert_equal(S.j[:], np.array([2, 5]))
    # N_incoming and N_outgoing should also be correct
    assert all(S.N_outgoing[2, :] == 2)
    assert all(S.N_incoming[:, 2] == 1)
    assert all(S.N_incoming[:, 5] == 1)

    assert len(S2) == 2 * len(SG2), str(len(S2))
    assert all(S2.v_pre[:] > 2)
    assert len(S3) == 5 * len(SG1), '%s != %s ' % (len(S3), 5 * len(SG1))
    assert all(S3.v_post[:] < 25)

    assert len(S4) == 2 * len(SG2), str(len(S4))
    assert all(S4.v_post[:] > 2)
    assert len(S5) == 5 * len(SG1), '%s != %s ' % (len(53), 5 * len(SG1))
    assert all(S5.v_pre[:] < 25)


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_synapse_creation_generator():
    G1 = NeuronGroup(10, 'v:1', threshold='False')
    G2 = NeuronGroup(20, 'v:1', threshold='False')
    G1.v = 'i'
    G2.v = '10 + i'
    SG1 = G1[:5]
    SG2 = G2[10:]
    S = Synapses(SG1, SG2, 'w:1', on_pre='v+=w')
    S.connect(j='i*2 + k for k in range(2)')  # diverging connections

    # connect based on pre-/postsynaptic state variables
    S2 = Synapses(SG1, SG2, 'w:1', on_pre='v+=w')
    S2.connect(j='k for k in range(N_post) if v_pre > 2')

    S3 = Synapses(SG1, SG2, 'w:1', on_pre='v+=w')
    S3.connect(j='k for k in range(N_post) if v_post < 25')

    S4 = Synapses(SG2, SG1, 'w:1', on_pre='v+=w')
    S4.connect(j='k for k in range(N_post) if v_post > 2')

    S5 = Synapses(SG2, SG1, 'w:1', on_pre='v+=w')
    S5.connect(j='k for k in range(N_post) if v_pre < 25')

    run(0*ms)  # for standalone

    # Internally, the "real" neuron indices should be used
    assert_equal(S._synaptic_pre[:], np.arange(5).repeat(2))
    assert_equal(S._synaptic_post[:], np.arange(10)+10)
    # For the user, the subgroup-relative indices should be presented
    assert_equal(S.i[:], np.arange(5).repeat(2))
    assert_equal(S.j[:], np.arange(10))

    # N_incoming and N_outgoing should also be correct
    assert all(S.N_outgoing[:] == 2)
    assert all(S.N_incoming[:] == 1)

    assert len(S2) == 2 * len(SG2), str(len(S2))
    assert all(S2.v_pre[:] > 2)
    assert len(S3) == 5 * len(SG1), '%s != %s ' % (len(S3), 5 * len(SG1))
    assert all(S3.v_post[:] < 25)

    assert len(S4) == 2 * len(SG2), str(len(S4))
    assert all(S4.v_post[:] > 2)
    assert len(S5) == 5 * len(SG1), '%s != %s ' % (len(S5), 5 * len(SG1))
    assert all(S5.v_pre[:] < 25)


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_synapse_creation_generator_multiple_synapses():
    G1 = NeuronGroup(10, 'v:1', threshold='False')
    G2 = NeuronGroup(20, 'v:1', threshold='False')
    G1.v = 'i'
    G2.v = '10 + i'
    SG1 = G1[:5]
    SG2 = G2[10:]
    S1 = Synapses(SG1, SG2, 'w:1', on_pre='v+=w')
    S1.connect(j='k for k in range(N_post)', n='i')

    S2 = Synapses(SG1, SG2, 'w:1', on_pre='v+=w')
    S2.connect(j='k for k in range(N_post)', n='j')

    S3 = Synapses(SG2, SG1, 'w:1', on_pre='v+=w')
    S3.connect(j='k for k in range(N_post)', n='i')

    S4 = Synapses(SG2, SG1, 'w:1', on_pre='v+=w')
    S4.connect(j='k for k in range(N_post)', n='j')

    S5 = Synapses(SG1, SG2, 'w:1', on_pre='v+=w')
    S5.connect(j='k for k in range(N_post)', n='i+j')

    S6 = Synapses(SG2, SG1, 'w:1', on_pre='v+=w')
    S6.connect(j='k for k in range(N_post)', n='i+j')

    S7 = Synapses(SG1, SG2, 'w:1', on_pre='v+=w')
    S7.connect(j='k for k in range(N_post)', n='int(v_pre>2)*2')

    S8 = Synapses(SG2, SG1, 'w:1', on_pre='v+=w')
    S8.connect(j='k for k in range(N_post)', n='int(v_post>2)*2')

    S9 = Synapses(SG1, SG2, 'w:1', on_pre='v+=w')
    S9.connect(j='k for k in range(N_post)', n='int(v_post>22)*2')

    S10 = Synapses(SG2, SG1, 'w:1', on_pre='v+=w')
    S10.connect(j='k for k in range(N_post)', n='int(v_pre>22)*2')

    run(0*ms)  # for standalone

    # straightforward loop instead of doing something clever...
    for source in xrange(len(SG1)):
        assert_equal(S1.j[source, :], np.arange(len(SG2)).repeat(source))
        assert_equal(S2.j[source, :], np.arange(len(SG2)).repeat(np.arange(len(SG2))))
        assert_equal(S3.i[:, source], np.arange(len(SG2)).repeat(np.arange(len(SG2))))
        assert_equal(S4.i[:, source], np.arange(len(SG2)).repeat(source))
        assert_equal(S5.j[source, :], np.arange(len(SG2)).repeat(np.arange(len(SG2))+source))
        assert_equal(S6.i[:, source], np.arange(len(SG2)).repeat(np.arange(len(SG2)) + source))
        if source > 2:
            assert_equal(S7.j[source, :], np.arange(len(SG2)).repeat(2))
            assert_equal(S8.i[:, source], np.arange(len(SG2)).repeat(2))
        else:
            assert len(S7.j[source, :]) == 0
            assert len(S8.i[:, source]) == 0
        assert_equal(S9.j[source, :], np.arange(3, len(SG2)).repeat(2))
        assert_equal(S10.i[:, source], np.arange(3, len(SG2)).repeat(2))


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_synapse_creation_generator_complex_ranges():
    G1 = NeuronGroup(10, 'v:1', threshold='False')
    G2 = NeuronGroup(20, 'v:1', threshold='False')
    G1.v = 'i'
    G2.v = '10 + i'
    SG1 = G1[:5]
    SG2 = G2[10:]
    S = Synapses(SG1, SG2, 'w:1', on_pre='v+=w')
    S.connect(j='i+k for k in range(N_post-i)')  # Connect to all j>i

    # connect based on pre-/postsynaptic state variables
    S2 = Synapses(SG1, SG2, 'w:1', on_pre='v+=w')
    S2.connect(j='k for k in range(N_post * int(v_pre > 2))')

    # connect based on pre-/postsynaptic state variables
    S3 = Synapses(SG2, SG1, 'w:1', on_pre='v+=w')
    S3.connect(j='k for k in range(N_post * int(v_pre > 22))')

    run(0*ms)  # for standalone

    for syn_source in xrange(5):
        # Internally, the "real" neuron indices should be used
        assert_equal(S._synaptic_post[syn_source, :],
                     10 + syn_source + np.arange(10 - syn_source))
        # For the user, the subgroup-relative indices should be presented
        assert_equal(S.j[syn_source, :], syn_source + np.arange(10-syn_source))

    assert len(S2) == 2 * len(SG2), str(len(S2))
    assert all(S2.v_pre[:] > 2)
    assert len(S3) == 7 * len(SG1), str(len(S3))
    assert all(S3.v_pre[:] > 22)


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_synapse_creation_generator_random():
    G1 = NeuronGroup(10, 'v:1', threshold='False')
    G2 = NeuronGroup(20, 'v:1', threshold='False')
    G1.v = 'i'
    G2.v = '10 + i'
    SG1 = G1[:5]
    SG2 = G2[10:]

    # connect based on pre-/postsynaptic state variables
    S2 = Synapses(SG1, SG2, 'w:1', on_pre='v+=w')
    S2.connect(j='k for k in sample(N_post, p=1.0*int(v_pre > 2))')

    S3 = Synapses(SG2, SG1, 'w:1', on_pre='v+=w')
    S3.connect(j='k for k in sample(N_post, p=1.0*int(v_pre > 22))')

    run(0*ms)  # for standalone

    assert len(S2) == 2 * len(SG2), str(len(S2))
    assert all(S2.v_pre[:] > 2)
    assert len(S3) == 7 * len(SG1), str(len(S3))
    assert all(S3.v_pre[:] > 22)


def test_synapse_access():
    G1 = NeuronGroup(10, 'v:1', threshold='False')
    G1.v = 'i'
    G2 = NeuronGroup(20, 'v:1', threshold='False')
    G2.v = 'i'
    SG1 = G1[:5]
    SG2 = G2[10:]
    S = Synapses(SG1, SG2, 'w:1', on_pre='v+=w')
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
    S = Synapses(G1, G2, 'w:1')
    S.connect()
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
    S = Synapses(G1, G2, 'w:1')
    S.connect()

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
@with_setup(teardown=reinit_devices)
def test_synaptic_propagation():
    G1 = NeuronGroup(10, 'v:1', threshold='v>1', reset='v=0')
    G1.v['i%2==1'] = 1.1 # odd numbers should spike
    G2 = NeuronGroup(20, 'v:1')
    SG1 = G1[1:6]
    SG2 = G2[10:]
    S = Synapses(SG1, SG2, on_pre='v+=1')
    S.connect('i==j')
    run(defaultclock.dt)
    expected = np.zeros(len(G2))
    # Neurons 1, 3, 5 spiked and are connected to 10, 12, 14
    expected[[10, 12, 14]] = 1
    assert_equal(np.asarray(G2.v).flatten(), expected)


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_synaptic_propagation_2():
    # This tests for the bug in github issue #461
    source = NeuronGroup(100, '', threshold='True')
    sub_source = source[99:]
    target = NeuronGroup(1, 'v:1')
    syn = Synapses(sub_source, target, on_pre='v+=1')
    syn.connect()
    run(defaultclock.dt)
    assert target.v[0] == 1.0


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_spike_monitor():
    G = NeuronGroup(10, 'v:1', threshold='v>1', reset='v=0')
    G.v[0] = 1.1
    G.v[2] = 1.1
    G.v[5] = 1.1
    SG = G[3:]
    SG2 = G[:3]
    s_mon = SpikeMonitor(G)
    sub_s_mon = SpikeMonitor(SG)
    sub_s_mon2 = SpikeMonitor(SG2)
    run(defaultclock.dt)
    assert_equal(s_mon.i, np.array([0, 2, 5]))
    assert_equal(s_mon.t_, np.zeros(3))
    assert_equal(sub_s_mon.i, np.array([2]))
    assert_equal(sub_s_mon.t_, np.zeros(1))
    assert_equal(sub_s_mon2.i, np.array([0, 2]))
    assert_equal(sub_s_mon2.t_, np.zeros(2))
    expected = np.zeros(10, dtype=int)
    expected[[0, 2, 5]] = 1
    assert_equal(s_mon.count, expected)
    expected = np.zeros(7, dtype=int)
    expected[[2]] = 1
    assert_equal(sub_s_mon.count, expected)
    assert_equal(sub_s_mon2.count, np.array([1, 0, 1]))


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
@with_setup(teardown=reinit_devices)
def test_no_reference_2():
    '''
    Using subgroups without keeping an explicit reference. Monitors
    '''
    G = NeuronGroup(2, 'v:1', threshold='v>1', reset='v=0')
    G.v = [0, 1.1]
    state_mon = StateMonitor(G[:1], 'v', record=True)
    spike_mon = SpikeMonitor(G[1:])
    rate_mon = PopulationRateMonitor(G[:2])
    run(2*defaultclock.dt)
    assert_equal(state_mon[0].v[:], np.zeros(2))
    assert_equal(spike_mon.i[:], np.array([0]))
    assert_equal(spike_mon.t[:], np.array([0])*second)
    assert_equal(rate_mon.rate[:], np.array([0.5, 0])/defaultclock.dt)


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_no_reference_3():
    '''
    Using subgroups without keeping an explicit reference. Monitors
    '''
    G = NeuronGroup(2, 'v:1', threshold='v>1', reset='v=0')
    G.v = [1.1, 0]
    S = Synapses(G[:1], G[1:], on_pre='v+=1')
    S.connect()
    run(defaultclock.dt)
    assert_equal(G.v[:], np.array([0, 1]))


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_no_reference_4():
    '''
    Using subgroups without keeping an explicit reference. Synapses
    '''
    G1 = NeuronGroup(10, 'v:1', threshold='v>1', reset='v=0')
    G1.v['i%2==1'] = 1.1 # odd numbers should spike
    G2 = NeuronGroup(20, 'v:1')
    S = Synapses(G1[1:6], G2[10:], on_pre='v+=1')
    S.connect('i==j')
    run(defaultclock.dt)
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
    test_state_variables_simple()
    test_state_variables_string_indices()
    test_state_variables_group_as_index()
    test_state_variables_group_as_index_problematic()
    test_state_monitor()
    test_shared_variable()
    test_synapse_creation()
    test_synapse_creation_generator()
    test_synapse_creation_generator_complex_ranges()
    test_synapse_creation_generator_random()
    test_synapse_creation_generator_multiple_synapses()
    test_synapse_access()
    test_synapses_access_subgroups()
    test_synapses_access_subgroups_problematic()
    test_subexpression_references()
    test_subexpression_no_references()
    test_synaptic_propagation()
    test_synaptic_propagation_2()
    test_spike_monitor()
    test_wrong_indexing()
    test_no_reference_1()
    test_no_reference_2()
    test_no_reference_3()
    test_no_reference_4()
    test_recursive_subgroup()
