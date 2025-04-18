import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_equal

from brian2 import *
from brian2.core.clocks import EventClock
from brian2.utils.logger import catch_logs


@pytest.mark.codegen_independent
def test_clock_attributes():
    clock = Clock(dt=1 * ms)
    assert_array_equal(clock.t, 0 * second)
    assert_array_equal(clock.timestep, 0)
    assert_array_equal(clock.dt, 1 * ms)


@pytest.mark.codegen_independent
def test_clock_dt_change():
    clock = Clock(dt=1 * ms)
    # at time 0s, all dt changes should be allowed
    clock.dt = 0.75 * ms
    clock._set_t_update_dt()
    clock.dt = 2.5 * ms
    clock._set_t_update_dt()
    clock.dt = 1 * ms
    clock._set_t_update_dt()

    # at 0.1ms only changes that are still representable as an integer of the
    # current time 1s are allowed
    clock.dt = 0.1 * ms
    clock._set_t_update_dt()

    clock.dt = 0.05 * ms
    clock._set_t_update_dt(target_t=0.1 * ms)
    clock.dt = 0.1 * ms
    clock._set_t_update_dt(target_t=0.1 * ms)
    clock.dt = 0.3 * ms
    with pytest.raises(ValueError):
        clock._set_t_update_dt(target_t=0.1 * ms)


@pytest.mark.codegen_independent
def test_defaultclock():
    defaultclock.dt = 1 * ms
    assert_equal(defaultclock.dt, 1 * ms)
    assert defaultclock.name == "defaultclock"


@pytest.mark.codegen_independent
def test_set_interval_warning():
    clock = Clock(dt=0.1 * ms)
    with catch_logs() as logs:
        clock.set_interval(0 * second, 1000 * second)  # no problem
    assert len(logs) == 0
    with catch_logs() as logs:
        clock.set_interval(0 * second, 10000000000 * second)  # too long
    assert len(logs) == 1
    assert logs[0][1].endswith("many_timesteps")


@pytest.mark.codegen_independent
def test_event_clock():
    times = [0.0, 0.1, 0.2, 0.3]
    event_clock = EventClock(times)

    assert_equal(event_clock.variables["t"].get_value(), 0.0)
    assert_equal(event_clock[1], 0.1)

    event_clock.advance()
    assert_equal(event_clock.variables["timestep"].get_value(), 1)
    assert_equal(event_clock.variables["t"].get_value(), 0.1)

    event_clock.set_interval(0.1 * second, 0.3 * second)
    assert_equal(event_clock.variables["timestep"].get_value(), 1)
    assert_equal(event_clock.variables["t"].get_value(), 0.1)


@pytest.mark.codegen_independent
def test_combined_clocks_with_run_at():
    # Create a simple NeuronGroup
    G = NeuronGroup(1, "v : 1")
    G.v = 0

    # Regular clock monitoring
    regular_mon = StateMonitor(G, "v", record=0, dt=1 * ms)

    # Define specific times for events
    event_times = [0.5 * ms, 2.5 * ms, 4.5 * ms]

    # Create an array to store event times
    event_values = []

    # Function to record values at specific times
    @network_operation(when="start")
    def record_event_values():
        # Store current time if it's one of our event times
        t = defaultclock.t
        for event_time in event_times:
            if abs(float(t - event_time)) < 1e-10:
                event_values.append(float(t))

    # Use run_at to modify v at specific times
    G.run_at("v += 1", times=event_times)

    # Run simulation
    run(5 * ms)

    # Verify regular monitoring worked
    assert_array_equal(regular_mon.t, np.arange(0, 5.1, 1) * ms)

    # Check events occurred at correct times
    assert_equal(len(event_values), len(event_times))
    assert_allclose(event_values, [float(t) for t in event_times])

    # Check the v values reflect the events
    expected_v = np.zeros_like(regular_mon.v[0])
    for t in event_times:
        time_idx = np.searchsorted(regular_mon.t, t)
        # All times at or after an event should have v increased
        if time_idx < len(expected_v):
            expected_v[time_idx:] += 1

    assert_array_equal(regular_mon.v[0], expected_v)


if __name__ == "__main__":
    test_clock_attributes()
    restore_initial_state()
    test_clock_dt_change()
    restore_initial_state()
    test_defaultclock()
    test_set_interval_warning()
    test_event_clock()
    test_combined_clocks_with_run_at()
