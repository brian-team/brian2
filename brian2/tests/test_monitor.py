import uuid
import tempfile
import logging

from numpy.testing.utils import assert_allclose, assert_array_equal, assert_raises
from nose import with_setup
from nose.plugins.attrib import attr

from brian2 import *
from brian2.devices.device import reinit_devices, set_device, reset_device
from brian2.utils.logger import catch_logs


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_spike_monitor():
    G = NeuronGroup(3, '''dv/dt = rate : 1
                          rate: Hz''', threshold='v>1', reset='v=0')
    # We don't use 100 and 1000Hz, because then the membrane potential would
    # be exactly at 1 after 10 resp. 100 timesteps. Due to floating point
    # issues this will not be exact,
    G.rate = [101, 0, 1001] * Hz

    mon = SpikeMonitor(G)

    assert_raises(ValueError, lambda: SpikeMonitor(G, order=1))  # need to specify 'when' as well
    # Creating a SpikeMonitor for a Synapses object should not work
    S = Synapses(G, G, on_pre='v += 0')
    S.connect()
    assert_raises(TypeError, lambda: SpikeMonitor(S))

    run(10*ms)

    spike_trains = mon.spike_trains()

    assert_allclose(mon.t[mon.i == 0], [9.9]*ms)
    assert len(mon.t[mon.i == 1]) == 0
    assert_allclose(mon.t[mon.i == 2], np.arange(10)*ms + 0.9*ms)
    assert_allclose(mon.t_[mon.i == 0], np.array([9.9*float(ms)]))
    assert len(mon.t_[mon.i == 1]) == 0
    assert_allclose(mon.t_[mon.i == 2], (np.arange(10) + 0.9)*float(ms))
    assert_allclose(spike_trains[0], [9.9]*ms)
    assert len(spike_trains[1]) == 0
    assert_allclose(spike_trains[2], np.arange(10)*ms + 0.9*ms)
    assert_array_equal(mon.count, np.array([1, 0, 10]))

    i, t = mon.it
    i_, t_ = mon.it_
    assert_array_equal(i, mon.i)
    assert_array_equal(i, i_)
    assert_array_equal(t, mon.t)
    assert_array_equal(t_, mon.t_)

    assert_raises(KeyError, lambda: spike_trains[3])
    assert_raises(KeyError, lambda: spike_trains[-1])
    assert_raises(KeyError, lambda: spike_trains['string'])


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_spike_monitor_variables():
    G = NeuronGroup(3, '''dv/dt = rate : 1
                          rate : Hz
                          prev_spikes : integer''',
                    threshold='v>1', reset='v=0; prev_spikes += 1')
    # We don't use 100 and 1000Hz, because then the membrane potential would
    # be exactly at 1 after 10 resp. 100 timesteps. Due to floating point
    # issues this will not be exact,
    G.rate = [101, 0, 1001] * Hz
    mon1 = SpikeMonitor(G, variables='prev_spikes')
    mon2 = SpikeMonitor(G, variables='prev_spikes', when='after_resets')
    run(10*ms)
    all_values = mon1.all_values()
    prev_spikes_values = mon1.values('prev_spikes')
    assert_array_equal(mon1.prev_spikes[mon1.i == 0], [0])
    assert_array_equal(prev_spikes_values[0], [0])
    assert_array_equal(all_values['prev_spikes'][0], [0])
    assert_array_equal(mon1.prev_spikes[mon1.i == 1], [])
    assert_array_equal(prev_spikes_values[1], [])
    assert_array_equal(all_values['prev_spikes'][1], [])
    assert_array_equal(mon1.prev_spikes[mon1.i == 2], np.arange(10))
    assert_array_equal(prev_spikes_values[2], np.arange(10))
    assert_array_equal(all_values['prev_spikes'][2], np.arange(10))
    assert_array_equal(mon2.prev_spikes[mon2.i == 0], [1])
    assert_array_equal(mon2.prev_spikes[mon2.i == 1], [])
    assert_array_equal(mon2.prev_spikes[mon2.i == 2], np.arange(10)+1)

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_spike_monitor_get_states():
    G = NeuronGroup(3, '''dv/dt = rate : 1
                          rate : Hz
                          prev_spikes : integer''',
                    threshold='v>1', reset='v=0; prev_spikes += 1')
    # We don't use 100 and 1000Hz, because then the membrane potential would
    # be exactly at 1 after 10 resp. 100 timesteps. Due to floating point
    # issues this will not be exact,
    G.rate = [101, 0, 1001] * Hz
    mon1 = SpikeMonitor(G, variables='prev_spikes')
    run(10*ms)
    all_states = mon1.get_states()
    assert set(all_states.keys()) == {'count', 'i', 't', 'prev_spikes', 'N'}
    assert_array_equal(all_states['count'], mon1.count[:])
    assert_array_equal(all_states['i'], mon1.i[:])
    assert_array_equal(all_states['t'], mon1.t[:])
    assert_array_equal(all_states['prev_spikes'], mon1.prev_spikes[:])
    assert_array_equal(all_states['N'], mon1.N)


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_event_monitor():
    G = NeuronGroup(3, '''dv/dt = rate : 1
                          rate: Hz''', events={'my_event': 'v>1'})
    G.run_on_event('my_event', 'v=0')
    # We don't use 100 and 1000Hz, because then the membrane potential would
    # be exactly at 1 after 10 resp. 100 timesteps. Due to floating point
    # issues this will not be exact,
    G.rate = [101, 0, 1001] * Hz

    mon = EventMonitor(G, 'my_event')
    net = Network(G, mon)
    net.run(10*ms)

    event_trains = mon.event_trains()

    assert_allclose(mon.t[mon.i == 0], [9.9]*ms)
    assert len(mon.t[mon.i == 1]) == 0
    assert_allclose(mon.t[mon.i == 2], np.arange(10)*ms + 0.9*ms)
    assert_allclose(mon.t_[mon.i == 0], np.array([9.9*float(ms)]))
    assert len(mon.t_[mon.i == 1]) == 0
    assert_allclose(mon.t_[mon.i == 2], (np.arange(10) + 0.9)*float(ms))
    assert_allclose(event_trains[0], [9.9]*ms)
    assert len(event_trains[1]) == 0
    assert_allclose(event_trains[2], np.arange(10)*ms + 0.9*ms)
    assert_array_equal(mon.count, np.array([1, 0, 10]))

    i, t = mon.it
    i_, t_ = mon.it_
    assert_array_equal(i, mon.i)
    assert_array_equal(i, i_)
    assert_array_equal(t, mon.t)
    assert_array_equal(t_, mon.t_)

    assert_raises(KeyError, lambda: event_trains[3])
    assert_raises(KeyError, lambda: event_trains[-1])
    assert_raises(KeyError, lambda: event_trains['string'])


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_event_monitor_no_record():
    # Check that you can switch off recording spike times/indices
    G = NeuronGroup(3, '''dv/dt = rate : 1
                          rate: Hz''', events={'my_event': 'v>1'},
                    threshold='v>1', reset='v=0')
    # We don't use 100 and 1000Hz, because then the membrane potential would
    # be exactly at 1 after 10 resp. 100 timesteps. Due to floating point
    # issues this will not be exact,
    G.rate = [101, 0, 1001] * Hz

    event_mon = EventMonitor(G, 'my_event', record=False)
    event_mon2 = EventMonitor(G, 'my_event', variables='rate', record=False)
    spike_mon = SpikeMonitor(G, record=False)
    spike_mon2 = SpikeMonitor(G, variables='rate', record=False)
    net = Network(G, event_mon, event_mon2, spike_mon, spike_mon2)
    net.run(10*ms)

    # i and t should not be there
    assert 'i' not in event_mon.variables
    assert 't' not in event_mon.variables
    assert 'i' not in spike_mon.variables
    assert 't' not in spike_mon.variables

    assert_array_equal(event_mon.count, np.array([1, 0, 10]))
    assert_array_equal(spike_mon.count, np.array([1, 0, 10]))
    assert spike_mon.num_spikes == sum(spike_mon.count)
    assert event_mon.num_events == sum(event_mon.count)

    # Other variables should still have been recorded
    assert len(spike_mon2.rate) == spike_mon.num_spikes
    assert len(event_mon2.rate) == event_mon.num_events


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
@with_setup(teardown=reinit_devices)
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
    no_record = StateMonitor(G, 'v', record=False)

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

    run(10*ms)

    # Check time recordings
    assert_array_equal(nothing_mon.t, v_mon.t)
    assert_array_equal(nothing_mon.t_, np.asarray(nothing_mon.t))
    assert_array_equal(nothing_mon.t_, v_mon.t_)
    assert_allclose(nothing_mon.t,
                    np.arange(len(nothing_mon.t)) * defaultclock.dt)
    assert_array_equal(no_record.t, v_mon.t)

    # Check v recording
    assert_allclose(v_mon.v.T,
                    np.exp(np.tile(-v_mon.t, (2, 1)).T / (10*ms)))
    assert_allclose(v_mon.v_.T,
                    np.exp(np.tile(-v_mon.t_, (2, 1)).T / float(10*ms)))
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

    assert all(all_mon[0].not_refractory[:] == True), ('not_refractory should '
                                                       'be True, but got'
                                                       '(not_refractory, v):'
                                                       '%s ' % str((all_mon.not_refractory,
                                                                    all_mon.v)))

    # Synapses
    assert_allclose(synapse_mon.w[:], np.tile(S.j[:]*nS,
                                              (synapse_mon.w[:].shape[1], 1)).T)

@with_setup(teardown=reinit_devices)
def test_state_monitor_record_single_timestep():
    G = NeuronGroup(1, 'dv/dt = -v/(5*ms) : 1')
    G.v = 1
    mon = StateMonitor(G, 'v', record=True)
    # Recording before a run should not work
    assert_raises(TypeError, lambda: mon.record_single_timestep())
    run(0.5*ms)
    assert mon.t[0] == 0*ms
    assert mon[0].v[0] == 1
    assert_allclose(mon.t[-1], 0.5*ms-defaultclock.dt)
    assert len(mon.t) == 5
    mon.record_single_timestep()
    assert_allclose(mon.t[-1], 0.5*ms)
    assert len(mon.t) == 6
    assert mon[0].v[-1] == G.v


@attr('cpp_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_state_monitor_record_single_timestep_cpp_standalone():
    set_device('cpp_standalone', build_on_run=False)
    G = NeuronGroup(1, 'dv/dt = -v/(5*ms) : 1')
    G.v = 1
    mon = StateMonitor(G, 'v', record=True)
    # Recording before a run should not work
    assert_raises(TypeError, lambda: mon.record_single_timestep())
    run(0.5*ms)
    mon.record_single_timestep()
    tempdir = tempfile.mkdtemp()
    device.build(directory=tempdir, compile=True, run=True,
                 with_output=False)
    assert_allclose(mon.t[-1], 0.5*ms)
    assert len(mon.t) == 6
    assert mon[0].v[-1] == G.v
    reset_device()

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_state_monitor_indexing():
    # Check indexing semantics
    G = NeuronGroup(10, 'v:volt')
    G.v = np.arange(10) * volt
    mon = StateMonitor(G, 'v', record=[5, 6, 7])

    run(2 * defaultclock.dt)

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

    assert_allclose(mon.t[1:], Quantity([defaultclock.dt]))

    assert_raises(IndexError, lambda: mon[8])
    assert_raises(TypeError, lambda: mon['string'])
    assert_raises(TypeError, lambda: mon[5.0])
    assert_raises(TypeError, lambda: mon[[5.0, 6.0]])

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_state_monitor_get_states():
    G = NeuronGroup(2, '''dv/dt = -v / (10*ms) : 1
                          f = clip(v, 0.1, 0.9) : 1
                          rate: Hz''', threshold='v>1', reset='v=0',
                    refractory=2*ms)
    G.rate = [100, 1000] * Hz
    G.v = 1
    mon = StateMonitor(G, ['v', 'f', 'rate'], record=True)
    run(10*defaultclock.dt)
    all_states = mon.get_states()
    assert set(all_states.keys()) == {'v', 'f', 'rate', 't', 'N'}
    assert_array_equal(all_states['v'].T, mon.v[:])
    assert_array_equal(all_states['f'].T, mon.f[:])
    assert_array_equal(all_states['rate'].T, mon.rate[:])
    assert_array_equal(all_states['t'], mon.t[:])
    assert_array_equal(all_states['N'], mon.N)

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_state_monitor_resize():
    # Test for issue #518 (weave/cython did not resize the Variable object)
    G = NeuronGroup(2, 'v : 1')
    mon = StateMonitor(G, 'v', record=True)
    defaultclock.dt = 0.1*ms
    run(1*ms)
    # On standalone, the size information of the variables is only updated
    # after the variable has been accessed, so we can not only check the size
    # information of the variables object
    assert len(mon.t) == 10
    assert mon.v.shape == (2, 10)
    assert mon.variables['t'].size == 10
    # Note that the internally stored variable has the transposed shape of the
    # variable that is visible to the user
    assert mon.variables['v'].size == (10, 2)

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_rate_monitor_1():
    G = NeuronGroup(5, 'v : 1', threshold='v>1') # no reset
    G.v = 1.1 # All neurons spike every time step
    rate_mon = PopulationRateMonitor(G)
    run(10*defaultclock.dt)

    assert_allclose(rate_mon.t, np.arange(10) * defaultclock.dt)
    assert_allclose(rate_mon.t_, np.arange(10) * float(defaultclock.dt))
    assert_allclose(rate_mon.rate, np.ones(10) / defaultclock.dt)
    assert_allclose(rate_mon.rate_, np.asarray(np.ones(10) / defaultclock.dt))

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_rate_monitor_2():
    G = NeuronGroup(10, 'v : 1', threshold='v>1') # no reset
    G.v['i<5'] = 1.1  # Half of the neurons fire every time step
    rate_mon = PopulationRateMonitor(G)
    net = Network(G, rate_mon)
    net.run(10*defaultclock.dt)
    assert_allclose(rate_mon.rate, 0.5 * np.ones(10) / defaultclock.dt)
    assert_allclose(rate_mon.rate_, 0.5 *np.asarray(np.ones(10) / defaultclock.dt))

@attr('codegen-independent')
def test_rate_monitor_smoothed_rate():
    # Test the filter response by having a single spiking neuron
    G = SpikeGeneratorGroup(1, [0], [1]*ms)
    r_mon = PopulationRateMonitor(G)
    run(3*ms)
    index = int(np.round(1*ms/defaultclock.dt))
    except_index = np.array([idx for idx in xrange(len(r_mon.rate))
                             if idx != index])
    assert_array_equal(r_mon.rate[except_index], 0*Hz)
    assert_allclose(r_mon.rate[index], 1/defaultclock.dt)
    ### Flat window
    # Using a flat window of size = dt should not change anything
    assert_allclose(r_mon.rate, r_mon.smooth_rate(window='flat', width=defaultclock.dt))
    smoothed = r_mon.smooth_rate(window='flat', width=5*defaultclock.dt)
    assert_array_equal(smoothed[:index-2], 0*Hz)
    assert_array_equal(smoothed[index+3:], 0*Hz)
    assert_allclose(smoothed[index-2:index+3], 0.2/defaultclock.dt)
    with catch_logs(log_level=logging.INFO):
        smoothed2 = r_mon.smooth_rate(window='flat', width=5.4*defaultclock.dt)
        assert_array_equal(smoothed, smoothed2)

    ### Gaussian window
    width = 5*defaultclock.dt
    smoothed = r_mon.smooth_rate(window='gaussian', width=width)
    # 0 outside of window
    assert_array_equal(smoothed[:index-10], 0*Hz)
    assert_array_equal(smoothed[index+11:], 0*Hz)
    # Gaussian around the spike
    gaussian = np.exp(-(r_mon.t[index-10:index+11] - 1*ms)**2/(2*width**2))
    gaussian /= sum(gaussian)
    assert_allclose(smoothed[index-10:index+11], 1/defaultclock.dt*gaussian)

    ### Arbitrary window
    window = np.ones(5)
    smoothed_flat = r_mon.smooth_rate(window='flat', width=5*defaultclock.dt)
    smoothed_custom = r_mon.smooth_rate(window=window)
    assert_allclose(smoothed_flat, smoothed_custom)


@attr('codegen-independent')
def test_rate_monitor_smoothed_rate_incorrect():
    # Test the filter response by having a single spiking neuron
    G = SpikeGeneratorGroup(1, [0], [1]*ms)
    r_mon = PopulationRateMonitor(G)
    run(2*ms)

    assert_raises(TypeError, lambda: r_mon.smooth_rate(window='flat'))  # no width
    assert_raises(TypeError, lambda: r_mon.smooth_rate(window=np.ones(5), width=1*ms))
    assert_raises(NotImplementedError, lambda: r_mon.smooth_rate(window='unknown', width=1*ms))
    assert_raises(TypeError, lambda: r_mon.smooth_rate(window=object()))
    assert_raises(TypeError, lambda: r_mon.smooth_rate(window=np.ones(5, 2)))
    assert_raises(TypeError, lambda: r_mon.smooth_rate(window=np.ones(4)))  # even number


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_rate_monitor_get_states():
    G = NeuronGroup(5, 'v : 1', threshold='v>1') # no reset
    G.v = 1.1 # All neurons spike every time step
    rate_mon = PopulationRateMonitor(G)
    run(10*defaultclock.dt)
    all_states = rate_mon.get_states()
    assert set(all_states.keys()) == {'rate', 't', 'N'}
    assert_array_equal(all_states['rate'], rate_mon.rate[:])
    assert_array_equal(all_states['t'], rate_mon.t[:])
    assert_array_equal(all_states['N'], rate_mon.N)


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_rate_monitor_subgroups():
    old_dt = defaultclock.dt
    defaultclock.dt = 0.01*ms
    G = NeuronGroup(4, '''dv/dt = rate : 1
                          rate : Hz''', threshold='v>1', reset='v=0')
    G.rate = [100, 200, 400, 800] * Hz
    rate_all = PopulationRateMonitor(G)
    rate_1 = PopulationRateMonitor(G[:2])
    rate_2 = PopulationRateMonitor(G[2:])
    run(10*ms)
    assert_allclose(mean(G.rate[:]), mean(rate_all.rate[:]))
    assert_allclose(mean(G.rate[:2]), mean(rate_1.rate[:]))
    assert_allclose(mean(G.rate[2:]), mean(rate_2.rate[:]))

    defaultclock.dt = old_dt


if __name__ == '__main__':
    test_spike_monitor()
    test_spike_monitor_get_states()
    test_spike_monitor_variables()
    test_event_monitor()
    test_event_monitor_no_record()
    test_state_monitor()
    test_state_monitor_record_single_timestep()
    test_state_monitor_record_single_timestep_cpp_standalone()
    test_state_monitor_get_states()
    test_state_monitor_indexing()
    test_state_monitor_resize()
    test_rate_monitor_1()
    test_rate_monitor_2()
    test_rate_monitor_smoothed_rate()
    test_rate_monitor_smoothed_rate_incorrect()
    test_rate_monitor_get_states()
    test_rate_monitor_subgroups()
