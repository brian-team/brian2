import pytest
from numpy.testing import assert_array_equal, assert_equal

from brian2 import *
from brian2.core.clocks import EventClock
from brian2.tests.test_network import NameLister
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
    times = [0.0 * ms, 0.1 * ms, 0.2 * ms, 0.3 * ms]
    event_clock = EventClock(times)

    assert_equal(event_clock.variables["t"].get_value(), 0.0)
    assert_equal(event_clock[1], 0.1 * ms)

    event_clock.advance()
    assert_equal(event_clock.variables["timestep"].get_value(), 1)
    assert_equal(event_clock.variables["t"].get_value(), 0.0001)

    event_clock.set_interval(0.1 * ms, 0.3 * ms)
    assert_equal(event_clock.variables["timestep"].get_value(), 1)
    assert_equal(event_clock.variables["t"].get_value(), 0.0001)


@pytest.mark.codegen_independent
def test_combined_clocks_with_run_at():

    # Reset updates
    NameLister.updates[:] = []

    # Regular NameLister at 1ms interval
    regular_lister = NameLister(name="x", dt=1 * ms, order=0)

    # Event NameLister at specific times
    event_times = [0.5 * ms, 2.5 * ms, 4 * ms]
    event_lister = NameLister(name="y", clock=EventClock(times=event_times), order=1)

    # Create and run the network
    net = Network(regular_lister, event_lister)
    net.run(5 * ms)

    # Get update string
    updates = "".join(NameLister.updates)

    # Expected output: "x" at 0,1,2,3,4ms = 5 times
    # "y" at 0.5, 2.5, 4.0ms = 3 times
    # We don't care about exact timing here, just the sequence
    expected_x_count = 5
    expected_y_count = 3

    x_count = updates.count("x")
    y_count = updates.count("y")

    assert (
        x_count == expected_x_count
    ), f"Expected {expected_x_count} x's, got {x_count}"
    assert (
        y_count == expected_y_count
    ), f"Expected {expected_y_count} y's, got {y_count}"

    # Optional: check full string if needed
    print(updates)


if __name__ == "__main__":
    test_clock_attributes()
    restore_initial_state()
    test_clock_dt_change()
    restore_initial_state()
    test_defaultclock()
    test_set_interval_warning()
    test_event_clock()
    test_combined_clocks_with_run_at()
