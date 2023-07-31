import logging
import tempfile
import uuid

import pytest
from numpy.testing import assert_array_equal

from brian2 import *
from brian2.devices.cpp_standalone.device import CPPStandaloneDevice
from brian2.tests.utils import assert_allclose
from brian2.utils.logger import catch_logs


@pytest.mark.standalone_compatible
def test_spike_monitor():
    G_without_threshold = NeuronGroup(5, "x : 1")
    G = NeuronGroup(
        3,
        """
        dv/dt = rate : 1
        rate: Hz
        """,
        threshold="v>1",
        reset="v=0",
    )
    # We don't use 100 and 1000Hz, because then the membrane potential would
    # be exactly at 1 after 10 resp. 100 timesteps. Due to floating point
    # issues this will not be exact,
    G.rate = [101, 0, 1001] * Hz

    mon = SpikeMonitor(G)

    with pytest.raises(ValueError):
        SpikeMonitor(G, order=1)  # need to specify 'when' as well
    with pytest.raises(ValueError) as ex:
        SpikeMonitor(G_without_threshold)
    assert "threshold" in str(ex)

    # Creating a SpikeMonitor for a Synapses object should not work
    S = Synapses(G, G, on_pre="v += 0")
    S.connect()
    with pytest.raises(TypeError):
        SpikeMonitor(S)

    run(10 * ms)

    spike_trains = mon.spike_trains()

    assert_allclose(mon.t[mon.i == 0], [9.9] * ms)
    assert len(mon.t[mon.i == 1]) == 0
    assert_allclose(mon.t[mon.i == 2], np.arange(10) * ms + 0.9 * ms)
    assert_allclose(mon.t_[mon.i == 0], np.array([9.9 * float(ms)]))
    assert len(mon.t_[mon.i == 1]) == 0
    assert_allclose(mon.t_[mon.i == 2], (np.arange(10) + 0.9) * float(ms))
    assert_allclose(spike_trains[0], [9.9] * ms)
    assert len(spike_trains[1]) == 0
    assert_allclose(spike_trains[2], np.arange(10) * ms + 0.9 * ms)
    assert_array_equal(mon.count, np.array([1, 0, 10]))

    i, t = mon.it
    i_, t_ = mon.it_
    assert_array_equal(i, mon.i)
    assert_array_equal(i, i_)
    assert_array_equal(t, mon.t)
    assert_array_equal(t_, mon.t_)

    with pytest.raises(KeyError):
        spike_trains[3]
    with pytest.raises(KeyError):
        spike_trains[-1]
    with pytest.raises(KeyError):
        spike_trains["string"]

    # Check that indexing into the VariableView works (this fails if we do not
    # update the N variable correctly)
    assert_allclose(mon.t[:5], [0.9, 1.9, 2.9, 3.9, 4.9] * ms)


def test_spike_monitor_indexing():
    generator = SpikeGeneratorGroup(3, [0, 0, 0, 1], [0, 1, 2, 1] * ms)
    mon = SpikeMonitor(generator)
    run(3 * ms)

    assert_array_equal(mon.t["i == 1"], [1] * ms)
    assert_array_equal(mon.t[mon.i == 1], [1] * ms)
    assert_array_equal(mon.i[mon.t > 1.5 * ms], [0] * ms)
    assert_array_equal(mon.i["t > 1.5 * ms"], [0] * ms)


@pytest.mark.standalone_compatible
def test_spike_monitor_variables():
    G = NeuronGroup(
        3,
        """
        dv/dt = rate : 1
       rate : Hz
       prev_spikes : integer
       """,
        threshold="v>1",
        reset="v=0; prev_spikes += 1",
    )
    # We don't use 100 and 1000Hz, because then the membrane potential would
    # be exactly at 1 after 10 resp. 100 timesteps. Due to floating point
    # issues this will not be exact,
    G.rate = [101, 0, 1001] * Hz
    mon1 = SpikeMonitor(G, variables="prev_spikes")
    mon2 = SpikeMonitor(G, variables="prev_spikes", when="after_resets")
    run(10 * ms)
    all_values = mon1.all_values()
    prev_spikes_values = mon1.values("prev_spikes")
    assert_array_equal(mon1.prev_spikes[mon1.i == 0], [0])
    assert_array_equal(prev_spikes_values[0], [0])
    assert_array_equal(all_values["prev_spikes"][0], [0])
    assert_array_equal(mon1.prev_spikes[mon1.i == 1], [])
    assert_array_equal(prev_spikes_values[1], [])
    assert_array_equal(all_values["prev_spikes"][1], [])
    assert_array_equal(mon1.prev_spikes[mon1.i == 2], np.arange(10))
    assert_array_equal(prev_spikes_values[2], np.arange(10))
    assert_array_equal(all_values["prev_spikes"][2], np.arange(10))
    assert_array_equal(mon2.prev_spikes[mon2.i == 0], [1])
    assert_array_equal(mon2.prev_spikes[mon2.i == 1], [])
    assert_array_equal(mon2.prev_spikes[mon2.i == 2], np.arange(10) + 1)


@pytest.mark.standalone_compatible
def test_spike_monitor_get_states():
    G = NeuronGroup(
        3,
        """dv/dt = rate : 1
                          rate : Hz
                          prev_spikes : integer""",
        threshold="v>1",
        reset="v=0; prev_spikes += 1",
    )
    # We don't use 100 and 1000Hz, because then the membrane potential would
    # be exactly at 1 after 10 resp. 100 timesteps. Due to floating point
    # issues this will not be exact,
    G.rate = [101, 0, 1001] * Hz
    mon1 = SpikeMonitor(G, variables="prev_spikes")
    run(10 * ms)
    all_states = mon1.get_states()
    assert set(all_states.keys()) == {"count", "i", "t", "prev_spikes", "N"}
    assert_array_equal(all_states["count"], mon1.count[:])
    assert_array_equal(all_states["i"], mon1.i[:])
    assert_array_equal(all_states["t"], mon1.t[:])
    assert_array_equal(all_states["prev_spikes"], mon1.prev_spikes[:])
    assert_array_equal(all_states["N"], mon1.N)


@pytest.mark.standalone_compatible
def test_spike_monitor_subgroups():
    G = NeuronGroup(6, """do_spike : boolean""", threshold="do_spike")
    G.do_spike = [True, False, False, False, True, True]
    spikes_all = SpikeMonitor(G)
    spikes_1 = SpikeMonitor(G[:2])
    spikes_2 = SpikeMonitor(G[2:4])
    spikes_3 = SpikeMonitor(G[4:])
    run(defaultclock.dt)
    assert_allclose(spikes_all.i, [0, 4, 5])
    assert_allclose(spikes_all.t, [0, 0, 0] * ms)
    assert_allclose(spikes_1.i, [0])
    assert_allclose(spikes_1.t, [0] * ms)
    assert len(spikes_2.i) == 0
    assert len(spikes_2.t) == 0
    assert_allclose(spikes_3.i, [0, 1])  # recorded spike indices are relative
    assert_allclose(spikes_3.t, [0, 0] * ms)


def test_spike_monitor_bug_824():
    # See github issue #824
    if prefs.codegen.target != "numpy":
        pytest.skip("numpy-only test")

    G = NeuronGroup(6, """do_spike : boolean""", threshold="do_spike")
    G.do_spike = [True, False, False, True, False, False]
    spikes_1 = SpikeMonitor(G[:3])
    spikes_2 = SpikeMonitor(G[3:])
    run(4 * defaultclock.dt)
    assert_array_equal(spikes_1.count, [4, 0, 0])
    assert_array_equal(spikes_2.count, [4, 0, 0])


@pytest.mark.standalone_compatible
def test_event_monitor():
    G = NeuronGroup(
        3,
        """
        dv/dt = rate : 1
        rate: Hz
        """,
        events={"my_event": "v>1"},
    )
    G.run_on_event("my_event", "v=0")
    # We don't use 100 and 1000Hz, because then the membrane potential would
    # be exactly at 1 after 10 resp. 100 timesteps. Due to floating point
    # issues this will not be exact,
    G.rate = [101, 0, 1001] * Hz

    mon = EventMonitor(G, "my_event")
    net = Network(G, mon)
    net.run(10 * ms)

    event_trains = mon.event_trains()

    assert_allclose(mon.t[mon.i == 0], [9.9] * ms)
    assert len(mon.t[mon.i == 1]) == 0
    assert_allclose(mon.t[mon.i == 2], np.arange(10) * ms + 0.9 * ms)
    assert_allclose(mon.t_[mon.i == 0], np.array([9.9 * float(ms)]))
    assert len(mon.t_[mon.i == 1]) == 0
    assert_allclose(mon.t_[mon.i == 2], (np.arange(10) + 0.9) * float(ms))
    assert_allclose(event_trains[0], [9.9] * ms)
    assert len(event_trains[1]) == 0
    assert_allclose(event_trains[2], np.arange(10) * ms + 0.9 * ms)
    assert_array_equal(mon.count, np.array([1, 0, 10]))

    i, t = mon.it
    i_, t_ = mon.it_
    assert_array_equal(i, mon.i)
    assert_array_equal(i, i_)
    assert_array_equal(t, mon.t)
    assert_array_equal(t_, mon.t_)

    with pytest.raises(KeyError):
        event_trains[3]
    with pytest.raises(KeyError):
        event_trains[-1]
    with pytest.raises(KeyError):
        event_trains["string"]


@pytest.mark.standalone_compatible
def test_event_monitor_no_record():
    # Check that you can switch off recording spike times/indices
    G = NeuronGroup(
        3,
        """
        dv/dt = rate : 1
        rate: Hz
        """,
        events={"my_event": "v>1"},
        threshold="v>1",
        reset="v=0",
    )
    # We don't use 100 and 1000Hz, because then the membrane potential would
    # be exactly at 1 after 10 resp. 100 timesteps. Due to floating point
    # issues this will not be exact,
    G.rate = [101, 0, 1001] * Hz

    event_mon = EventMonitor(G, "my_event", record=False)
    event_mon2 = EventMonitor(G, "my_event", variables="rate", record=False)
    spike_mon = SpikeMonitor(G, record=False)
    spike_mon2 = SpikeMonitor(G, variables="rate", record=False)
    net = Network(G, event_mon, event_mon2, spike_mon, spike_mon2)
    net.run(10 * ms)

    # i and t should not be there
    assert "i" not in event_mon.variables
    assert "t" not in event_mon.variables
    assert "i" not in spike_mon.variables
    assert "t" not in spike_mon.variables

    assert_array_equal(event_mon.count, np.array([1, 0, 10]))
    assert_array_equal(spike_mon.count, np.array([1, 0, 10]))
    assert spike_mon.num_spikes == sum(spike_mon.count)
    assert event_mon.num_events == sum(event_mon.count)

    # Other variables should still have been recorded
    assert len(spike_mon2.rate) == spike_mon.num_spikes
    assert len(event_mon2.rate) == event_mon.num_events


@pytest.mark.standalone_compatible
def test_spike_trains():
    # An arbitrary combination of indices that has been shown to sort in an
    # unstable way with quicksort and therefore lead to out-of-order values
    # in the dictionary returned by spike_trains() of several neurons (see #725)
    generator = SpikeGeneratorGroup(
        10,
        [6, 1, 2, 4, 6, 9, 1, 4, 7, 8, 0, 6, 3, 6, 4, 8, 9, 2, 5, 3],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] * ms,
        dt=1 * ms,
    )
    monitor = SpikeMonitor(generator)
    run(19.1 * ms)
    trains = monitor.spike_trains()
    for idx, spike_times in trains.items():
        assert all(
            np.diff(spike_times) > 0 * ms
        ), f"Spike times for neuron {int(idx)} are not sorted"


def test_synapses_state_monitor():
    G = NeuronGroup(2, "")
    S = Synapses(G, G, "w: siemens")
    S.connect(True)
    S.w = "j*nS"

    # record from a Synapses object (all synapses connecting to neuron 1)
    synapse_mon = StateMonitor(S, "w", record=S[:, 1])
    synapse_mon2 = StateMonitor(S, "w", record=S["j==1"])

    net = Network(G, S, synapse_mon, synapse_mon2)
    net.run(10 * ms)
    # Synaptic variables
    assert_allclose(synapse_mon[S[0, 1]].w, 1 * nS)
    assert_allclose(synapse_mon.w[1], 1 * nS)
    assert_allclose(synapse_mon2[S[0, 1]].w, 1 * nS)
    assert_allclose(synapse_mon2[S["i==0 and j==1"]].w, 1 * nS)
    assert_allclose(synapse_mon2.w[1], 1 * nS)


@pytest.mark.standalone_compatible
def test_state_monitor():
    # Unique name to get the warning even for repeated runs of the test
    unique_name = f"neurongroup_{str(uuid.uuid4()).replace('-', '_')}"
    # Check that all kinds of variables can be recorded
    G = NeuronGroup(
        2,
        """
        dv/dt = -v / (10*ms) : 1
        f = clip(v, 0.1, 0.9) : 1
        rate: Hz
        """,
        threshold="v>1",
        reset="v=0",
        refractory=2 * ms,
        name=unique_name,
    )
    G.rate = [100, 1000] * Hz
    G.v = 1

    S = Synapses(G, G, "w: siemens")
    S.connect(True)
    S.w = "j*nS"

    # A bit peculiar, but in principle one should be allowed to record
    # nothing except for the time
    nothing_mon = StateMonitor(G, [], record=True)
    no_record = StateMonitor(G, "v", record=False)

    # Use a single StateMonitor
    v_mon = StateMonitor(G, "v", record=True)
    v_mon1 = StateMonitor(G, "v", record=[1])

    # Use a StateMonitor for specified variables
    multi_mon = StateMonitor(G, ["v", "f", "rate"], record=True)
    multi_mon1 = StateMonitor(G, ["v", "f", "rate"], record=[1])

    # Use a StateMonitor recording everything
    all_mon = StateMonitor(G, True, record=True)

    # Record synapses with explicit indices (the only way allowed in standalone)
    synapse_mon = StateMonitor(S, "w", record=np.arange(len(G) ** 2))

    run(10 * ms)

    # Check time recordings
    assert_array_equal(nothing_mon.t, v_mon.t)
    assert_array_equal(nothing_mon.t_, np.asarray(nothing_mon.t))
    assert_array_equal(nothing_mon.t_, v_mon.t_)
    assert_allclose(nothing_mon.t, np.arange(len(nothing_mon.t)) * defaultclock.dt)
    assert_array_equal(no_record.t, v_mon.t)

    # Check v recording
    assert_allclose(v_mon.v.T, np.exp(np.tile(-v_mon.t, (2, 1)).T / (10 * ms)))
    assert_allclose(v_mon.v_.T, np.exp(np.tile(-v_mon.t_, (2, 1)).T / float(10 * ms)))
    assert_array_equal(v_mon.v, multi_mon.v)
    assert_array_equal(v_mon.v_, multi_mon.v_)
    assert_array_equal(v_mon.v, all_mon.v)
    assert_array_equal(v_mon.v[1:2], v_mon1.v)
    assert_array_equal(multi_mon.v[1:2], multi_mon1.v)
    assert len(no_record.v) == 0

    # Other variables
    assert_array_equal(
        multi_mon.rate_.T, np.tile(np.atleast_2d(G.rate_), (multi_mon.rate.shape[1], 1))
    )
    assert_array_equal(multi_mon.rate[1:2], multi_mon1.rate)
    assert_allclose(np.clip(multi_mon.v, 0.1, 0.9), multi_mon.f)
    assert_allclose(np.clip(multi_mon1.v, 0.1, 0.9), multi_mon1.f)

    assert all(
        all_mon[0].not_refractory[:] == True
    ), "not_refractory should be True, but got(not_refractory, v):%s " % str(
        (all_mon.not_refractory, all_mon.v)
    )

    # Synapses
    assert_allclose(
        synapse_mon.w[:], np.tile(S.j[:] * nS, (synapse_mon.w[:].shape[1], 1)).T
    )


@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_state_monitor_record_single_timestep():
    G = NeuronGroup(1, "dv/dt = -v/(5*ms) : 1")
    G.v = 1
    mon = StateMonitor(G, "v", record=True)
    # Recording before a run should not work
    with pytest.raises(TypeError):
        mon.record_single_timestep()
    run(0.5 * ms)
    mon.record_single_timestep()
    device.build(direct_call=False, **device.build_options)
    assert mon.t[0] == 0 * ms
    assert mon[0].v[0] == 1
    assert_allclose(mon.t[-1], 0.5 * ms)
    assert len(mon.t) == 6
    assert mon[0].v[-1] == G.v


@pytest.mark.standalone_compatible
def test_state_monitor_bigger_dt():
    # Check that it is possible to record with a slower clock, i.e. bigger dt
    G = NeuronGroup(1, "dv/dt = -v/(5*ms) : 1", method="exact")
    G.v = 1
    mon = StateMonitor(G, "v", record=True)
    slow_mon = StateMonitor(G, "v", record=True, dt=defaultclock.dt * 5)
    run(defaultclock.dt * 11)
    assert len(mon.t) == len(mon.v[0]) == 11
    assert len(slow_mon.t) == len(slow_mon.v[0]) == 3
    for timepoint in [0, 5, 10]:
        assert mon.t[timepoint] == slow_mon.t[timepoint // 5]
        assert mon.v[0, timepoint] == slow_mon.v[0, timepoint // 5]


@pytest.mark.standalone_compatible
def test_state_monitor_indexing():
    # Check indexing semantics
    G = NeuronGroup(10, "v:volt")
    G.v = np.arange(10) * volt
    mon = StateMonitor(G, "v", record=[5, 6, 7])

    run(2 * defaultclock.dt)

    assert_array_equal(mon.v, np.array([[5, 5], [6, 6], [7, 7]]) * volt)
    assert_array_equal(mon.v_, np.array([[5, 5], [6, 6], [7, 7]]))
    assert_array_equal(mon[5].v, mon.v[0])
    assert_array_equal(mon[7].v, mon.v[2])
    assert_array_equal(mon[[5, 7]].v, mon.v[[0, 2]])
    assert_array_equal(mon[np.array([5, 7])].v, mon.v[[0, 2]])

    assert_allclose(mon.t[1:], Quantity([defaultclock.dt]))

    with pytest.raises(IndexError):
        mon[8]
    with pytest.raises(TypeError):
        mon["string"]
    with pytest.raises(TypeError):
        mon[5.0]
    with pytest.raises(TypeError):
        mon[[5.0, 6.0]]


@pytest.mark.standalone_compatible
def test_state_monitor_get_states():
    G = NeuronGroup(
        2,
        """
        dv/dt = -v / (10*ms) : 1
        f = clip(v, 0.1, 0.9) : 1
        rate: Hz
        """,
        threshold="v>1",
        reset="v=0",
        refractory=2 * ms,
    )
    G.rate = [100, 1000] * Hz
    G.v = 1
    mon = StateMonitor(G, ["v", "f", "rate"], record=True)
    run(10 * defaultclock.dt)
    all_states = mon.get_states()
    assert set(all_states.keys()) == {"v", "f", "rate", "t", "N"}
    assert_array_equal(all_states["v"].T, mon.v[:])
    assert_array_equal(all_states["f"].T, mon.f[:])
    assert_array_equal(all_states["rate"].T, mon.rate[:])
    assert_array_equal(all_states["t"], mon.t[:])
    assert_array_equal(all_states["N"], mon.N)


@pytest.mark.standalone_compatible
def test_state_monitor_resize():
    # Test for issue #518 (cython did not resize the Variable object)
    G = NeuronGroup(2, "v : 1")
    mon = StateMonitor(G, "v", record=True)
    defaultclock.dt = 0.1 * ms
    run(1 * ms)
    # On standalone, the size information of the variables is only updated
    # after the variable has been accessed, so we can not only check the size
    # information of the variables object
    assert len(mon.t) == 10
    assert mon.v.shape == (2, 10)
    assert mon.variables["t"].size == 10
    # Note that the internally stored variable has the transposed shape of the
    # variable that is visible to the user
    assert mon.variables["v"].size == (10, 2)


@pytest.mark.standalone_compatible
def test_state_monitor_synapses():
    # Check that recording from synapses works correctly
    G = NeuronGroup(5, "v : 1", threshold="False", reset="v = 0")
    S1 = Synapses(G, G, "w : 1", on_pre="v_post += w")
    S1.run_regularly("w += 1")
    S1.connect(i=[0, 1], j=[2, 3])
    S1.w = "i"
    # We can check the record argument even in standalone mode, since we created
    # the synapses with an array of indices of known length
    with catch_logs() as l:
        S1_mon = StateMonitor(S1, "w", record=[0, 1])
    assert len(l) == 0

    with pytest.raises(IndexError):
        StateMonitor(S1, "w", record=[0, 2])

    S2 = Synapses(G, G, "w : 1", on_pre="v_post += w")
    S2.run_regularly("w += 1")
    S2.connect("i==j")  # Not yet executed for standalone mode
    S2.w = "i"
    with catch_logs() as l:
        S2_mon = StateMonitor(S2, "w", record=[0, 4])

    if isinstance(get_device(), CPPStandaloneDevice):
        assert len(l) == 1
        assert l[0][0] == "WARNING"
        assert l[0][1].endswith(".cannot_check_statemonitor_indices")
    else:
        assert len(l) == 0
    run(2 * defaultclock.dt)
    assert_array_equal(S1_mon.w[:], [[0, 1], [1, 2]])
    assert_array_equal(S2_mon.w[:], [[0, 1], [4, 5]])


@pytest.mark.codegen_independent
def test_statemonitor_namespace():
    # Make sure that StateMonitor is correctly inheriting its source's namespace
    G = NeuronGroup(2, "x = i + y : integer", namespace={"y": 3})
    mon = StateMonitor(G, "x", record=True)
    run(defaultclock.dt, namespace={})
    assert_array_equal(mon.x, [[3], [4]])


@pytest.mark.standalone_compatible
def test_rate_monitor_1():
    G = NeuronGroup(5, "v : 1", threshold="v>1")  # no reset
    G.v = 1.1  # All neurons spike every time step
    rate_mon = PopulationRateMonitor(G)
    run(10 * defaultclock.dt)

    assert_allclose(rate_mon.t, np.arange(10) * defaultclock.dt)
    assert_allclose(rate_mon.t_, np.arange(10) * defaultclock.dt_)
    assert_allclose(rate_mon.t, np.arange(10) * defaultclock.dt)
    assert_allclose(rate_mon.rate, np.ones(10) / defaultclock.dt)
    assert_allclose(rate_mon.rate_, np.asarray(np.ones(10) / defaultclock.dt_))
    # Check that indexing into the VariableView works (this fails if we do not
    # update the N variable correctly)
    assert_allclose(rate_mon.t[:5], np.arange(5) * defaultclock.dt)


@pytest.mark.standalone_compatible
def test_rate_monitor_2():
    G = NeuronGroup(10, "v : 1", threshold="v>1")  # no reset
    G.v["i<5"] = 1.1  # Half of the neurons fire every time step
    rate_mon = PopulationRateMonitor(G)
    net = Network(G, rate_mon)
    net.run(10 * defaultclock.dt)
    assert_allclose(rate_mon.rate, 0.5 * np.ones(10) / defaultclock.dt)
    assert_allclose(rate_mon.rate_, 0.5 * np.asarray(np.ones(10) / defaultclock.dt_))


@pytest.mark.codegen_independent
def test_rate_monitor_smoothed_rate():
    # Test the filter response by having a single spiking neuron
    G = SpikeGeneratorGroup(1, [0], [1] * ms)
    r_mon = PopulationRateMonitor(G)
    run(3 * ms)
    index = int(np.round(1 * ms / defaultclock.dt))
    except_index = np.array([idx for idx in range(len(r_mon.rate)) if idx != index])
    assert_array_equal(r_mon.rate[except_index], 0 * Hz)
    assert_allclose(r_mon.rate[index], 1 / defaultclock.dt)
    ### Flat window
    # Using a flat window of size = dt should not change anything
    assert_allclose(r_mon.rate, r_mon.smooth_rate(window="flat", width=defaultclock.dt))
    smoothed = r_mon.smooth_rate(window="flat", width=5 * defaultclock.dt)
    assert_array_equal(smoothed[: index - 2], 0 * Hz)
    assert_array_equal(smoothed[index + 3 :], 0 * Hz)
    assert_allclose(smoothed[index - 2 : index + 3], 0.2 / defaultclock.dt)
    with catch_logs(log_level=logging.INFO):
        smoothed2 = r_mon.smooth_rate(window="flat", width=5.4 * defaultclock.dt)
        assert_array_equal(smoothed, smoothed2)

    ### Gaussian window
    width = 5 * defaultclock.dt
    smoothed = r_mon.smooth_rate(window="gaussian", width=width)
    # 0 outside of window
    assert_array_equal(smoothed[: index - 10], 0 * Hz)
    assert_array_equal(smoothed[index + 11 :], 0 * Hz)
    # Gaussian around the spike
    gaussian = np.exp(
        -((r_mon.t[index - 10 : index + 11] - 1 * ms) ** 2) / (2 * width**2)
    )
    gaussian /= sum(gaussian)
    assert_allclose(smoothed[index - 10 : index + 11], 1 / defaultclock.dt * gaussian)

    ### Arbitrary window
    window = np.ones(5)
    smoothed_flat = r_mon.smooth_rate(window="flat", width=5 * defaultclock.dt)
    smoothed_custom = r_mon.smooth_rate(window=window)
    assert_allclose(smoothed_flat, smoothed_custom)


@pytest.mark.codegen_independent
def test_rate_monitor_smoothed_rate_incorrect():
    # Test the filter response by having a single spiking neuron
    G = SpikeGeneratorGroup(1, [0], [1] * ms)
    r_mon = PopulationRateMonitor(G)
    run(2 * ms)

    with pytest.raises(TypeError):
        r_mon.smooth_rate(window="flat")  # no width
    with pytest.raises(TypeError):
        r_mon.smooth_rate(window=np.ones(5), width=1 * ms)
    with pytest.raises(NotImplementedError):
        r_mon.smooth_rate(window="unknown", width=1 * ms)
    with pytest.raises(TypeError):
        r_mon.smooth_rate(window=object())
    with pytest.raises(TypeError):
        r_mon.smooth_rate(window=np.ones(5, 2))
    with pytest.raises(TypeError):
        r_mon.smooth_rate(window=np.ones(4))  # even number


@pytest.mark.standalone_compatible
def test_rate_monitor_get_states():
    G = NeuronGroup(5, "v : 1", threshold="v>1")  # no reset
    G.v = 1.1  # All neurons spike every time step
    rate_mon = PopulationRateMonitor(G)
    run(10 * defaultclock.dt)
    all_states = rate_mon.get_states()
    assert set(all_states.keys()) == {"rate", "t", "N"}
    assert_array_equal(all_states["rate"], rate_mon.rate[:])
    assert_array_equal(all_states["t"], rate_mon.t[:])
    assert_array_equal(all_states["N"], rate_mon.N)


@pytest.mark.standalone_compatible
def test_rate_monitor_subgroups():
    old_dt = defaultclock.dt
    defaultclock.dt = 0.01 * ms
    G = NeuronGroup(
        4,
        """
        dv/dt = rate : 1
        rate : Hz
        """,
        threshold="v>0.999",
        reset="v=0",
    )
    G.rate = [100, 200, 400, 800] * Hz
    rate_all = PopulationRateMonitor(G)
    rate_1 = PopulationRateMonitor(G[:2])
    rate_2 = PopulationRateMonitor(G[2:])
    run(1 * second)
    assert_allclose(mean(G.rate_[:]), mean(rate_all.rate_[:]))
    assert_allclose(mean(G.rate_[:2]), mean(rate_1.rate_[:]))
    assert_allclose(mean(G.rate_[2:]), mean(rate_2.rate_[:]))

    defaultclock.dt = old_dt


@pytest.mark.standalone_compatible
def test_rate_monitor_subgroups_2():
    G = NeuronGroup(6, """do_spike : boolean""", threshold="do_spike")
    G.do_spike = [True, False, False, False, True, True]
    rate_all = PopulationRateMonitor(G)
    rate_1 = PopulationRateMonitor(G[:2])
    rate_2 = PopulationRateMonitor(G[2:4])
    rate_3 = PopulationRateMonitor(G[4:])
    run(2 * defaultclock.dt)
    assert_allclose(rate_all.rate, 0.5 / defaultclock.dt)
    assert_allclose(rate_1.rate, 0.5 / defaultclock.dt)
    assert_allclose(rate_2.rate, 0 * Hz)
    assert_allclose(rate_3.rate, 1 / defaultclock.dt)


@pytest.mark.codegen_independent
def test_monitor_str_repr():
    # Basic test that string representations are not empty
    G = NeuronGroup(2, "dv/dt = -v/(10*ms) : 1", threshold="v>1", reset="v=0")
    spike_mon = SpikeMonitor(G)
    state_mon = StateMonitor(G, "v", record=True)
    rate_mon = PopulationRateMonitor(G)
    for obj in [spike_mon, state_mon, rate_mon]:
        assert len(str(obj))
        assert len(repr(obj))


if __name__ == "__main__":
    from _pytest.outcomes import Skipped

    test_spike_monitor()
    test_spike_monitor_indexing()
    test_spike_monitor_get_states()
    test_spike_monitor_subgroups()
    try:
        test_spike_monitor_bug_824()
    except Skipped:
        pass
    test_spike_monitor_variables()
    test_event_monitor()
    test_event_monitor_no_record()
    test_spike_trains()
    test_synapses_state_monitor()
    test_state_monitor()
    test_state_monitor_record_single_timestep()
    test_state_monitor_get_states()
    test_state_monitor_indexing()
    test_state_monitor_resize()
    test_rate_monitor_1()
    test_rate_monitor_2()
    test_rate_monitor_smoothed_rate()
    test_rate_monitor_smoothed_rate_incorrect()
    test_rate_monitor_get_states()
    test_rate_monitor_subgroups()
    test_rate_monitor_subgroups_2()
    test_monitor_str_repr()
