import uuid

from numpy.testing.utils import assert_allclose, assert_array_equal, assert_raises
from nose import with_setup
from nose.plugins.attrib import attr

from brian2 import *
from brian2.devices.device import restore_device
from brian2.utils.logger import catch_logs


@attr('standalone-compatible')
@with_setup(teardown=restore_device)
def test_spike_monitor():
    G = NeuronGroup(2, '''dv/dt = rate : 1
                          rate: Hz''', threshold='v>1', reset='v=0')
    # We don't use 100 and 1000Hz, because then the membrane potential would
    # be exactly at 1 after 10 resp. 100 timesteps. Due to floating point
    # issues this will not be exact,
    G.rate = [101, 1001] * Hz

    mon = SpikeMonitor(G)
    net = Network(G, mon)
    net.run(10*ms)

    assert_allclose(mon.t[mon.i == 0], [9.9]*ms)
    assert_allclose(mon.t[mon.i == 1], np.arange(10)*ms + 0.9*ms)
    assert_allclose(mon.t_[mon.i == 0], np.array([9.9*float(ms)]))
    assert_allclose(mon.t_[mon.i == 1], (np.arange(10) + 0.9)*float(ms))
    assert_array_equal(mon.count, np.array([1, 10]))

    i, t = mon.it
    i_, t_ = mon.it_
    assert_array_equal(i, mon.i)
    assert_array_equal(i, i_)
    assert_array_equal(t, mon.t)
    assert_array_equal(t_, mon.t_)


def test_synapses_state_monitor():
    G = NeuronGroup(2, '')
    S = Synapses(G, G, 'w: siemens')
    S.connect(True)
    S.w = 'j*nS'

    # record from a Synapses object (all synapses connecting to neuron 1)
    synapse_mon = StateMonitor(S, 'w', record=S[:, 1])
    synapse_mon2 = StateMonitor(S, 'w', record=S['j==1'])

    net = Network(G, S, synapse_mon, synapse_mon2)
    net.run(10*ms)
    # Synaptic variables
    assert_allclose(synapse_mon[S[0, 1]].w, 1*nS)
    assert_allclose(synapse_mon.w[1], 1*nS)
    assert_allclose(synapse_mon2[S[0, 1]].w, 1*nS)
    assert_allclose(synapse_mon2[S['i==0 and j==1']].w, 1*nS)
    assert_allclose(synapse_mon2.w[1], 1*nS)

@attr('standalone-compatible')
@with_setup(teardown=restore_device)
def test_state_monitor():
    # Unique name to get the warning even for repeated runs of the test
    unique_name = 'neurongroup_' + str(uuid.uuid4()).replace('-', '_')
    # Check that all kinds of variables can be recorded
    G = NeuronGroup(2, '''dv/dt = -v / (10*ms) : 1
                          f = clip(v, 0.1, 0.9) : 1
                          rate: Hz''', threshold='v>1', reset='v=0',
                    refractory=2*ms, name=unique_name)
    G.rate = [100, 1000] * Hz
    G.v = 1

    S = Synapses(G, G, 'w: siemens')
    S.connect(True)
    S.w = 'j*nS'

    # A bit peculiar, but in principle one should be allowed to record
    # nothing except for the time
    nothing_mon = StateMonitor(G, [], record=True)

    # A more common case is that the user forgets the record argument (which
    # defaults to ``None``) -- raise a warning in this case
    with catch_logs() as l:
        no_record = StateMonitor(G, 'v')
        assert len(l) == 1

    # Use a single StateMonitor
    v_mon = StateMonitor(G, 'v', record=True)
    v_mon1 = StateMonitor(G, 'v', record=[1])

    # Use a StateMonitor for specified variables
    multi_mon = StateMonitor(G, ['v', 'f', 'rate'], record=True)
    multi_mon1 = StateMonitor(G, ['v', 'f', 'rate'], record=[1])

    # Use a StateMonitor recording everything
    all_mon = StateMonitor(G, True, record=True)

    # Record synapses with explicit indices (the only way allowed in standalone)
    synapse_mon = StateMonitor(S, 'w', record=np.arange(len(G)**2))

    net = Network(G, S,
                  nothing_mon, no_record,
                  v_mon, v_mon1,
                  multi_mon, multi_mon1,
                  all_mon,
                  synapse_mon)
    net.run(10*ms)

    # Check time recordings
    assert_array_equal(nothing_mon.t, v_mon.t)
    assert_array_equal(nothing_mon.t_, np.asarray(nothing_mon.t))
    assert_array_equal(nothing_mon.t_, v_mon.t_)
    assert_allclose(nothing_mon.t,
                    np.arange(len(nothing_mon.t)) * defaultclock.dt)
    assert_array_equal(no_record.t, v_mon.t)

    # Check v recording
    assert_allclose(v_mon.v.T,
                    np.exp(np.tile(-v_mon.t - defaultclock.dt, (2, 1)).T / (10*ms)))
    assert_allclose(v_mon.v_.T,
                    np.exp(np.tile(-v_mon.t_ - defaultclock.dt_, (2, 1)).T / float(10*ms)))
    assert_array_equal(v_mon.v, multi_mon.v)
    assert_array_equal(v_mon.v_, multi_mon.v_)
    assert_array_equal(v_mon.v, all_mon.v)
    assert_array_equal(v_mon.v[1:2], v_mon1.v)
    assert_array_equal(multi_mon.v[1:2], multi_mon1.v)
    assert len(no_record.v) == 0

    # Other variables
    assert_array_equal(multi_mon.rate_.T, np.tile(np.atleast_2d(G.rate_),
                                         (multi_mon.rate.shape[1], 1)))
    assert_array_equal(multi_mon.rate[1:2], multi_mon1.rate)
    assert_allclose(np.clip(multi_mon.v, 0.1, 0.9), multi_mon.f)
    assert_allclose(np.clip(multi_mon1.v, 0.1, 0.9), multi_mon1.f)

    assert all(all_mon[0].not_refractory[:] == True)

    # Synapses
    assert_allclose(synapse_mon.w[:], np.tile(S.j[:]*nS,
                                              (synapse_mon.w[:].shape[1], 1)).T)

@attr('standalone-compatible')
@with_setup(teardown=restore_device)
def test_state_monitor_indexing():
    # Check indexing semantics
    G = NeuronGroup(10, 'v:volt')
    G.v = np.arange(10) * volt
    mon = StateMonitor(G, 'v', record=[5, 6, 7])

    net = Network(G, mon)
    net.run(2 * defaultclock.dt)

    assert_array_equal(mon.v, np.array([[5, 5],
                                  [6, 6],
                                  [7, 7]]) * volt)
    assert_array_equal(mon.v_, np.array([[5, 5],
                                   [6, 6],
                                   [7, 7]]))
    assert_array_equal(mon[5].v, mon.v[0])
    assert_array_equal(mon[7].v, mon.v[2])
    assert_array_equal(mon[[5, 7]].v, mon.v[[0, 2]])
    assert_array_equal(mon[np.array([5, 7])].v, mon.v[[0, 2]])

    assert_raises(IndexError, lambda: mon[8])
    assert_raises(TypeError, lambda: mon['string'])
    assert_raises(TypeError, lambda: mon[5.0])
    assert_raises(TypeError, lambda: mon[[5.0, 6.0]])

@attr('standalone-compatible')
@with_setup(teardown=restore_device)
def test_rate_monitor():
    G = NeuronGroup(5, 'v : 1', threshold='v>1') # no reset
    G.v = 1.1 # All neurons spike every time step
    rate_mon = PopulationRateMonitor(G)
    net = Network(G, rate_mon)
    net.run(10*defaultclock.dt)

    assert_allclose(rate_mon.t, np.arange(10) * defaultclock.dt)
    assert_allclose(rate_mon.t_, np.arange(10) * float(defaultclock.dt))
    assert_allclose(rate_mon.rate, np.ones(10) / defaultclock.dt)
    assert_allclose(rate_mon.rate_, np.asarray(np.ones(10) / defaultclock.dt))

    G = NeuronGroup(10, 'v : 1', threshold='v>1') # no reset
    G.v['i<5'] = 1.1  # Half of the neurons fire every time step
    rate_mon = PopulationRateMonitor(G)
    net = Network(G, rate_mon)
    net.run(10*defaultclock.dt)
    assert_allclose(rate_mon.rate, 0.5 * np.ones(10) / defaultclock.dt)
    assert_allclose(rate_mon.rate_, 0.5 *np.asarray(np.ones(10) / defaultclock.dt))

@attr('standalone-compatible')
@with_setup(teardown=restore_device)
def test_rate_monitor_subgroups():
    old_dt = defaultclock.dt
    defaultclock.dt = 0.01*ms
    G = NeuronGroup(4, '''dv/dt = rate : 1
                          rate : Hz''', threshold='v>1', reset='v=0')
    G.rate = [100, 200, 400, 800] * Hz
    rate_all = PopulationRateMonitor(G)
    rate_1 = PopulationRateMonitor(G[:2])
    rate_2 = PopulationRateMonitor(G[2:])
    s_mon = SpikeMonitor(G)
    net = Network(G, rate_all, rate_1, rate_2, s_mon)
    net.run(10*ms)
    assert_allclose(mean(G.rate[:]), mean(rate_all.rate[:]))
    assert_allclose(mean(G.rate[:2]), mean(rate_1.rate[:]))
    assert_allclose(mean(G.rate[2:]), mean(rate_2.rate[:]))

    defaultclock.dt = old_dt


if __name__ == '__main__':
    test_spike_monitor()
    test_state_monitor()
    test_state_monitor_indexing()
    test_rate_monitor()
    test_rate_monitor_subgroups()
