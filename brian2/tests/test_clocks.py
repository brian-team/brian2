import pytest
from numpy.testing import assert_array_equal, assert_equal

from brian2 import *
from brian2.core.clocks import EventClock
from brian2.tests.test_network import NameLister
from brian2.units.fundamentalunits import DimensionMismatchError
from brian2.utils.logger import catch_logs


@pytest.mark.codegen_independent
def test_clock_attributes():
    clock = Clock(dt=1 * ms)
    assert_array_equal(clock.t, 0 * second)
    assert_array_equal(clock.timestep, 0)
    assert_array_equal(clock.dt, 1 * ms)


@pytest.mark.codegen_independent
def test_clock_comparisons():
    clock1 = Clock(dt=1 * ms)
    clock2 = Clock(dt=2 * ms)
    event_clock1 = EventClock(times=[0 * ms, 1 * ms])
    event_clock2 = EventClock(times=[0 * ms, 2 * ms])

    clocks = [clock1, clock2, event_clock1, event_clock2]
    assert clock1.same_time(clock2) and clock2.same_time(clock1)
    assert clock1.same_time(event_clock1) and event_clock1.same_time(clock1)
    assert event_clock1.same_time(event_clock2) and event_clock2.same_time(event_clock1)

    for clock in clocks:
        clock.advance()
    assert clock1.same_time(event_clock1) and event_clock1.same_time(clock1)
    assert clock2.same_time(event_clock2) and event_clock2.same_time(clock2)
    assert not (clock1.same_time(clock2) or clock2.same_time(clock1))
    assert not (clock1.same_time(event_clock2) or event_clock2.same_time(clock1))
    assert not (
        event_clock1.same_time(event_clock2) or event_clock2.same_time(event_clock1)
    )


@pytest.mark.codegen_independent
def test_clock_set_interval():
    clock = Clock(dt=1 * ms)
    clock.set_interval(0 * ms, 2 * ms)
    assert clock.t == 0 * ms
    clock.advance()
    assert clock.t == 1 * ms
    clock.advance()
    assert clock.t == 2 * ms
    with pytest.raises(StopIteration):
        clock.advance()
    assert clock.t == 2 * ms


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
    times = [0.0 * ms, 0.3 * ms, 0.5 * ms, 0.6 * ms]
    event_clock = EventClock(times)

    assert_equal(event_clock.variables["t"].get_value(), 0.0 * ms)
    assert_equal(event_clock[1], 0.3 * ms)

    event_clock.advance()
    assert_equal(event_clock.variables["timestep"].get_value(), 1)
    assert_equal(event_clock.variables["t"].get_value(), 0.0003)

    event_clock.set_interval(0.3 * ms, 0.6 * ms)
    assert_equal(event_clock.variables["timestep"].get_value(), 1)
    assert_equal(event_clock.variables["t"].get_value(), 0.0003)
    event_clock.advance()
    event_clock.advance()

    with pytest.raises(StopIteration):
        event_clock.advance()

    invalid_times = [0.0 * volt, 0.5 * volt]
    with pytest.raises(DimensionMismatchError) as excinfo:
        EventClock(invalid_times)


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
    expected_x_count = 5
    expected_y_count = 3

    x_count = updates.count("x")
    y_count = updates.count("y")

    assert x_count == expected_x_count, (
        f"Expected {expected_x_count} x's, got {x_count}"
    )
    assert y_count == expected_y_count, (
        f"Expected {expected_y_count} y's, got {y_count}"
    )

    expected_output = "xyxxyxxy"
    assert updates == expected_output, f"Expected {expected_output}, got {updates}"


if __name__ == "__main__":
    test_clock_attributes()
    restore_initial_state()
    test_clock_dt_change()
    restore_initial_state()
    test_defaultclock()
    test_set_interval_warning()
    test_event_clock()
    test_combined_clocks_with_run_at()
