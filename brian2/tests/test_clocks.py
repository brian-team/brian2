from brian2 import *
from brian2.utils.logger import catch_logs
from numpy.testing import assert_raises, assert_equal, assert_array_equal
from nose import with_setup
from nose.plugins.attrib import attr


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_clock_attributes():
    clock = Clock(dt=1*ms)
    assert_array_equal(clock.t, 0*second)
    assert_array_equal(clock.timestep, 0)
    assert_array_equal(clock.dt, 1*ms)


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_clock_dt_change():
    clock = Clock(dt=1*ms)
    # at time 0s, all dt changes should be allowed
    clock.dt = 0.75*ms
    clock._set_t_update_dt()
    clock.dt = 2.5*ms
    clock._set_t_update_dt()
    clock.dt = 1*ms
    clock._set_t_update_dt()

    # at 0.1ms only changes that are still representable as an integer of the
    # current time 1s are allowed
    clock.dt = 0.1*ms
    clock._set_t_update_dt()

    clock.dt = 0.05*ms
    clock._set_t_update_dt(target_t=0.1*ms)
    clock.dt = 0.1*ms
    clock._set_t_update_dt(target_t=0.1*ms)
    clock.dt = 0.3*ms
    assert_raises(ValueError, lambda: clock._set_t_update_dt(target_t=0.1*ms))


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_defaultclock():
    defaultclock.dt = 1*ms
    assert_equal(defaultclock.dt, 1*ms)
    assert defaultclock.name == 'defaultclock'


@attr('codegen-independent')
def test_set_interval_warning():
    clock = Clock(dt=0.1*ms)
    with catch_logs() as logs:
        clock.set_interval(0*second, 1000*second)  # no problem
    assert len(logs) == 0
    with catch_logs() as logs:
        clock.set_interval(0*second, 10000000000*second)  # too long
    assert len(logs) == 1
    assert logs[0][1].endswith('many_timesteps')


@attr('codegen-independent')
def test_very_long_dt():
    # See github issue #1054
    clock1 = Clock(dt=0.1*ms)
    clock2 = Clock(dt=100000*second)
    clock1.set_interval(0*ms, 1*ms)
    clock2.set_interval(0*ms, 1*ms)  # The clock should advance
    assert clock1.timestep[:] == 0
    assert clock2.timestep[:] == 0
    assert clock1._i_end == 10
    assert clock2._i_end == 1
    # Simulate advancing the clock
    clock1.variables['timestep'].set_value(clock1._i_end)
    clock1.variables['t'].set_value(clock1._i_end * clock1.dt)
    clock2.variables['timestep'].set_value(clock2._i_end)
    clock2.variables['t'].set_value(clock2._i_end * clock2.dt)

    clock1.set_interval(1*ms, 2*ms)
    clock2.set_interval(1*ms, 2*ms)  # The clock should not advance
    assert clock1.timestep[:] == 10
    assert clock2.timestep[:] == 1
    assert clock1._i_end == 20
    assert clock2._i_end == 1


if __name__ == '__main__':
    test_clock_attributes()
    restore_initial_state()
    test_clock_dt_change()
    restore_initial_state()
    test_defaultclock()
    test_set_interval_warning()
    test_very_long_dt()
