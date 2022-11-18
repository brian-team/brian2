"""
Tests for `SpikeGeneratorGroup`
"""
import os
import tempfile

import pytest
from numpy.testing import assert_equal, assert_array_equal

from brian2 import *
from brian2.core.network import schedule_propagation_offset
from brian2.devices.device import reinit_and_delete
from brian2.tests.utils import assert_allclose, exc_isinstance
from brian2.utils.logger import catch_logs


@pytest.mark.standalone_compatible
def test_spikegenerator_connected():
    """
    Test that `SpikeGeneratorGroup` connects properly.
    """
    G = NeuronGroup(10, "v:1")
    mon = StateMonitor(G, "v", record=True, when="end")
    indices = np.array([3, 2, 1, 1, 4, 5])
    times = np.array([6, 5, 4, 3, 3, 1]) * ms
    SG = SpikeGeneratorGroup(10, indices, times)
    S = Synapses(SG, G, on_pre="v+=1")
    S.connect(j="i")
    run(7 * ms)
    # The following neurons should not receive any spikes
    for idx in [0, 6, 7, 8, 9]:
        assert all(mon[idx].v == 0)
    offset = schedule_propagation_offset()
    # The following neurons should receive a single spike
    for idx, time in zip([2, 3, 4, 5], [5, 6, 3, 1] * ms):
        assert all(mon[idx].v[mon.t < time + offset] == 0)
        assert all(mon[idx].v[mon.t >= time + offset] == 1)
    # This neuron receives two spikes
    assert all(mon[1].v[mon.t < 3 * ms + offset] == 0)
    assert all(mon[1].v[(mon.t >= 3 * ms + offset) & (mon.t < 4 * ms + offset)] == 1)
    assert all(mon[1].v[(mon.t >= 4 * ms + offset)] == 2)


@pytest.mark.standalone_compatible
def test_spikegenerator_basic():
    """
    Basic test for `SpikeGeneratorGroup`.
    """
    indices = np.array([3, 2, 1, 1, 2, 3, 3, 2, 1])
    times = np.array([1, 4, 4, 3, 2, 4, 2, 3, 2]) * ms
    SG = SpikeGeneratorGroup(5, indices, times)
    s_mon = SpikeMonitor(SG)
    run(5 * ms)
    _compare_spikes(5, indices, times, s_mon)


@pytest.mark.standalone_compatible
def test_spikegenerator_basic_sorted():
    """
    Basic test for `SpikeGeneratorGroup` with already sorted spike events.
    """
    indices = np.array([3, 1, 2, 3, 1, 2, 1, 2, 3])
    times = np.array([1, 2, 2, 2, 3, 3, 4, 4, 4]) * ms
    SG = SpikeGeneratorGroup(5, indices, times)
    s_mon = SpikeMonitor(SG)
    run(5 * ms)
    _compare_spikes(5, indices, times, s_mon)


@pytest.mark.standalone_compatible
def test_spikegenerator_basic_sorted_with_sorted():
    """
    Basic test for `SpikeGeneratorGroup` with already sorted spike events.
    """
    indices = np.array([3, 1, 2, 3, 1, 2, 1, 2, 3])
    times = np.array([1, 2, 2, 2, 3, 3, 4, 4, 4]) * ms
    SG = SpikeGeneratorGroup(5, indices, times, sorted=True)
    s_mon = SpikeMonitor(SG)
    run(5 * ms)
    _compare_spikes(5, indices, times, s_mon)


@pytest.mark.standalone_compatible
def test_spikegenerator_period():
    """
    Basic test for `SpikeGeneratorGroup`.
    """
    indices = np.array([3, 2, 1, 1, 2, 3, 3, 2, 1])
    times = np.array([1, 4, 4, 3, 2, 4, 2, 3, 2]) * ms
    SG = SpikeGeneratorGroup(5, indices, times, period=5 * ms)

    s_mon = SpikeMonitor(SG)
    run(10 * ms)
    for idx in range(5):
        generator_spikes = sorted(
            [(idx, time) for time in times[indices == idx]]
            + [(idx, time + 5 * ms) for time in times[indices == idx]]
        )
        recorded_spikes = sorted([(idx, time) for time in s_mon.t[s_mon.i == idx]])
        assert_allclose(generator_spikes, recorded_spikes)


@pytest.mark.codegen_independent
def test_spikegenerator_extreme_period():
    """
    Basic test for `SpikeGeneratorGroup`.
    """
    indices = np.array([0, 1, 2])
    times = np.array([0, 1, 2]) * ms
    SG = SpikeGeneratorGroup(5, indices, times, period=1e6 * second)
    s_mon = SpikeMonitor(SG)
    with catch_logs() as l:
        run(10 * ms)

    assert_equal(s_mon.i, np.array([0, 1, 2]))
    assert_allclose(s_mon.t, [0, 1, 2] * ms)
    assert len(l) == 1 and l[0][1].endswith("spikegenerator_long_period")


@pytest.mark.standalone_compatible
def test_spikegenerator_period_rounding():
    # See discussion in PR #1042
    # The last spike will be considered to be in the time step *after* 1s, due
    # to the way our rounding works. Although probably not what the user
    # expects, this should therefore raise an error. In previous versions of
    # Brian, this did not raise any error but silently discarded the spike.
    with pytest.raises(ValueError):
        SpikeGeneratorGroup(
            1, [0, 0, 0], [0 * ms, 0.9 * ms, 0.99999 * ms], period=1 * ms, dt=0.1 * ms
        )
    # This should also raise a ValueError, since the last two spikes fall into
    # the same bin
    s = SpikeGeneratorGroup(
        1, [0, 0, 0], [0 * ms, 0.9 * ms, 0.96 * ms], period=1 * ms, dt=0.1 * ms
    )
    net = Network(s)
    with pytest.raises(BrianObjectException) as exc:
        net.run(0 * ms)
    assert exc_isinstance(exc, ValueError)


def test_spikegenerator_period_repeat():
    """
    Basic test for `SpikeGeneratorGroup`.
    """
    indices = np.zeros(10)
    times = arange(0, 1, 0.1) * ms

    rec = np.rec.fromarrays([times, indices], names=["t", "i"])
    rec.sort()
    sorted_times = np.ascontiguousarray(rec.t) * 1000
    sorted_indices = np.ascontiguousarray(rec.i)
    SG = SpikeGeneratorGroup(1, indices, times, period=1 * ms)
    s_mon = SpikeMonitor(SG)
    net = Network(SG, s_mon)
    rate = PopulationRateMonitor(SG)
    for idx in range(5):
        net.run(1 * ms)
        assert (idx + 1) * len(SG.spike_time) == s_mon.num_spikes


def _compare_spikes(
    N, indices, times, recorded, start_time=0 * ms, end_time=1e100 * second
):
    for idx in range(N):
        generator_spikes = sorted([(idx, time) for time in times[indices == idx]])
        recorded_spikes = sorted(
            [
                (idx, time)
                for time in recorded.t[recorded.i == idx]
                if time >= start_time and time < end_time
            ]
        )
        assert_allclose(generator_spikes, recorded_spikes)


@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_spikegenerator_change_spikes():
    indices1 = np.array([3, 2, 1, 1, 2, 3, 3, 2, 1])
    times1 = np.array([1, 4, 4, 3, 2, 4, 2, 3, 2]) * ms
    SG = SpikeGeneratorGroup(5, indices1, times1)
    s_mon = SpikeMonitor(SG)
    net = Network(SG, s_mon)
    net.run(5 * ms)

    indices2 = np.array([3, 2, 1, 1, 2, 3, 3, 2, 1, 3, 3, 3, 1, 2])
    times2 = (
        np.array([1, 4, 4, 3, 2, 4, 2, 3, 2, 4.5, 4.7, 4.8, 4.5, 4.7]) * ms + 5 * ms
    )

    SG.set_spikes(indices2, times2)
    net.run(5 * ms)

    indices3 = np.array([4, 1, 0])
    times3 = np.array([1, 3, 4]) * ms + 10 * ms

    SG.set_spikes(indices3, times3)
    net.run(5 * ms)
    device.build(direct_call=False, **device.build_options)
    _compare_spikes(5, indices1, times1, s_mon, 0 * ms, 5 * ms)
    _compare_spikes(5, indices2, times2, s_mon, 5 * ms, 10 * ms)
    _compare_spikes(5, indices3, times3, s_mon, 10 * ms)


@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_spikegenerator_change_period():
    """
    Basic test for `SpikeGeneratorGroup`.
    """
    indices1 = np.array([3, 2, 1, 1, 2, 3, 3, 2, 1])
    times1 = np.array([1, 4, 4, 3, 2, 4, 2, 3, 2]) * ms
    SG = SpikeGeneratorGroup(5, indices1, times1, period=5 * ms)
    s_mon = SpikeMonitor(SG)
    net = Network(SG, s_mon)
    net.run(10 * ms)

    indices2 = np.array([3, 2, 1, 1, 2, 3, 3, 2, 1, 3, 3, 3, 1, 2])
    times2 = (
        np.array([1, 4, 4, 3, 2, 4, 2, 3, 2, 4.5, 4.7, 4.8, 4.5, 4.7]) * ms + 10 * ms
    )

    SG.set_spikes(indices2, times2)
    net.run(10 * ms)  # period should no longer be in effect
    device.build(direct_call=False, **device.build_options)

    _compare_spikes(
        5,
        np.hstack([indices1, indices1]),
        np.hstack([times1, times1 + 5 * ms]),
        s_mon,
        0 * ms,
        10 * ms,
    )
    _compare_spikes(5, indices2, times2, s_mon, 10 * ms)


@pytest.mark.codegen_independent
def test_spikegenerator_incorrect_values():
    with pytest.raises(TypeError):
        SpikeGeneratorGroup(0, [], [] * second)
    # Floating point value for N
    with pytest.raises(TypeError):
        SpikeGeneratorGroup(1.5, [], [] * second)
    # Negative index
    with pytest.raises(ValueError):
        SpikeGeneratorGroup(5, [0, 3, -1], [0, 1, 2] * ms)
    # Too high index
    with pytest.raises(ValueError):
        SpikeGeneratorGroup(5, [0, 5, 1], [0, 1, 2] * ms)
    # Negative time
    with pytest.raises(ValueError):
        SpikeGeneratorGroup(5, [0, 1, 2], [0, -1, 2] * ms)


@pytest.mark.codegen_independent
def test_spikegenerator_incorrect_period():
    """
    Test that you cannot provide incorrect period arguments or combine
    inconsistent period and dt arguments.
    """
    # Period is negative
    with pytest.raises(ValueError):
        SpikeGeneratorGroup(1, [], [] * second, period=-1 * ms)

    # Period is smaller than the highest spike time
    with pytest.raises(ValueError):
        SpikeGeneratorGroup(1, [0], [2] * ms, period=1 * ms)
    # Period is not an integer multiple of dt
    SG = SpikeGeneratorGroup(1, [], [] * second, period=1.25 * ms, dt=0.1 * ms)
    net = Network(SG)
    with pytest.raises(BrianObjectException) as exc:
        net.run(0 * ms)
    assert exc_isinstance(exc, NotImplementedError)

    SG = SpikeGeneratorGroup(1, [], [] * second, period=0.101 * ms, dt=0.1 * ms)
    net = Network(SG)
    with pytest.raises(BrianObjectException) as exc:
        net.run(0 * ms)
    assert exc_isinstance(exc, NotImplementedError)

    SG = SpikeGeneratorGroup(1, [], [] * second, period=3.333 * ms, dt=0.1 * ms)
    net = Network(SG)
    with pytest.raises(BrianObjectException) as exc:
        net.run(0 * ms)
    assert exc_isinstance(exc, NotImplementedError)

    # This should not raise an error (see #1041)
    SG = SpikeGeneratorGroup(1, [], [] * ms, period=150 * ms, dt=0.1 * ms)
    net = Network(SG)
    net.run(0 * ms)

    # Period is smaller than dt
    SG = SpikeGeneratorGroup(1, [], [] * second, period=1 * ms, dt=2 * ms)
    net = Network(SG)
    with pytest.raises(BrianObjectException) as exc:
        net.run(0 * ms)
    assert exc_isinstance(exc, ValueError)


def test_spikegenerator_rounding():
    # all spikes should fall into the first time bin
    indices = np.arange(100)
    times = np.linspace(0, 0.1, 100, endpoint=False) * ms
    SG = SpikeGeneratorGroup(100, indices, times, dt=0.1 * ms)
    mon = SpikeMonitor(SG)
    net = Network(SG, mon)
    net.run(0.1 * ms)
    assert_equal(mon.count, np.ones(100))

    # all spikes should fall in separate bins
    dt = 0.1 * ms
    indices = np.zeros(10000)
    times = np.arange(10000) * dt
    SG = SpikeGeneratorGroup(1, indices, times, dt=dt)
    target = NeuronGroup(
        1, "count : 1", threshold="True", reset="count=0"
    )  # set count to zero at every time step
    syn = Synapses(SG, target, on_pre="count+=1")
    syn.connect()
    mon = StateMonitor(target, "count", record=0, when="end")
    net = Network(SG, target, syn, mon)
    # change the schedule so that resets are processed before synapses
    net.schedule = ["start", "groups", "thresholds", "resets", "synapses", "end"]
    net.run(10000 * dt)
    assert_equal(mon[0].count, np.ones(10000))


@pytest.mark.standalone_compatible
@pytest.mark.long
def test_spikegenerator_rounding_long():
    # all spikes should fall in separate bins
    dt = 0.1 * ms
    N = 1000000
    indices = np.zeros(N)
    times = np.arange(N) * dt
    SG = SpikeGeneratorGroup(1, indices, times, dt=dt)
    target = NeuronGroup(1, "count : 1")
    syn = Synapses(SG, target, on_pre="count+=1")
    syn.connect()
    spikes = SpikeMonitor(SG)
    mon = StateMonitor(target, "count", record=0, when="end")
    run(N * dt, report="text")
    assert spikes.count[0] == N, f"expected {int(N)} spikes, got {int(spikes.count[0])}"
    assert all(np.diff(mon[0].count[:]) == 1)


@pytest.mark.standalone_compatible
@pytest.mark.long
def test_spikegenerator_rounding_period():
    # all spikes should fall in separate bins
    dt = 0.1 * ms
    N = 100
    repeats = 10000
    indices = np.zeros(N)
    times = np.arange(N) * dt
    SG = SpikeGeneratorGroup(1, indices, times, dt=dt, period=N * dt)
    target = NeuronGroup(1, "count : 1")
    syn = Synapses(SG, target, on_pre="count+=1")
    syn.connect()
    spikes = SpikeMonitor(SG)
    mon = StateMonitor(target, "count", record=0, when="end")
    run(N * repeats * dt, report="text")
    # print np.int_(np.round(spikes.t/dt))
    assert_equal(spikes.count[0], N * repeats)
    assert all(np.diff(mon[0].count[:]) == 1)


@pytest.mark.codegen_independent
def test_spikegenerator_multiple_spikes_per_bin():
    # Multiple spikes per bin are of course fine if they don't belong to the
    # same neuron
    SG = SpikeGeneratorGroup(2, [0, 1], [0, 0.05] * ms, dt=0.1 * ms)
    net = Network(SG)
    net.run(0 * ms)

    # This should raise an error
    SG = SpikeGeneratorGroup(2, [0, 0], [0, 0.05] * ms, dt=0.1 * ms)
    net = Network(SG)
    with pytest.raises(BrianObjectException) as exc:
        net.run(0 * ms)
    print(exc.value.__cause__)
    assert exc_isinstance(exc, ValueError)

    # More complicated scenario where dt changes between runs
    defaultclock.dt = 0.1 * ms
    SG = SpikeGeneratorGroup(2, [0, 0], [0.05, 0.15] * ms)
    net = Network(SG)
    net.run(0 * ms)  # all is fine
    defaultclock.dt = 0.2 * ms  # Now the two spikes fall into the same bin
    with pytest.raises(BrianObjectException) as exc:
        net.run(0 * ms)
    assert exc_isinstance(exc, ValueError)


@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_spikegenerator_multiple_runs():
    indices = np.zeros(5)
    times = np.arange(5) * ms
    spike_gen = SpikeGeneratorGroup(1, indices, times)  # all good
    spike_mon = SpikeMonitor(spike_gen)
    run(5 * ms)
    # Setting the same spike times again should not do anything, since they are
    # before the start of the current simulation
    spike_gen.set_spikes(indices, times)
    # however, a warning should be raised
    with catch_logs() as l:
        run(5 * ms)
        device.build(direct_call=False, **device.build_options)
    assert len(l) == 1 and l[0][1].endswith("ignored_spikes")
    assert spike_mon.num_spikes == 5


def test_spikegenerator_restore():
    # Check whether SpikeGeneratorGroup works with store/restore
    # See github issue #1084
    gen = SpikeGeneratorGroup(1, [0, 0, 0], [0, 1, 2] * ms)
    mon = SpikeMonitor(gen)
    store()
    run(3 * ms)
    assert_array_equal(mon.i, [0, 0, 0])
    assert_allclose(mon.t, [0, 1, 2] * ms)
    restore()
    run(3 * ms)
    assert_array_equal(mon.i, [0, 0, 0])
    assert_allclose(mon.t, [0, 1, 2] * ms)


if __name__ == "__main__":
    import time

    start = time.time()

    test_spikegenerator_connected()
    test_spikegenerator_basic()
    test_spikegenerator_basic_sorted()
    test_spikegenerator_basic_sorted_with_sorted()
    test_spikegenerator_period()
    test_spikegenerator_period_rounding()
    test_spikegenerator_extreme_period()
    test_spikegenerator_period_repeat()
    test_spikegenerator_change_spikes()
    test_spikegenerator_change_period()
    test_spikegenerator_incorrect_values()
    test_spikegenerator_incorrect_period()
    test_spikegenerator_rounding()
    test_spikegenerator_rounding_long()
    test_spikegenerator_rounding_period()
    test_spikegenerator_multiple_spikes_per_bin()
    test_spikegenerator_multiple_runs()
    test_spikegenerator_restore()
    print("Tests took", time.time() - start)
