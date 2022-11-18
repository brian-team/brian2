from brian2 import *
from brian2.utils.logger import catch_logs
from numpy.testing import assert_equal, assert_array_equal
import pytest


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


if __name__ == "__main__":
    test_clock_attributes()
    restore_initial_state()
    test_clock_dt_change()
    restore_initial_state()
    test_defaultclock()
    test_set_interval_warning()
