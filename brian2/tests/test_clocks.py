from brian2 import *
from numpy.testing import assert_raises, assert_equal
from nose import with_setup
from nose.plugins.attrib import attr

@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_clocks():
    clock = Clock(dt=1*ms)
    assert_equal(clock.t, 0*second)
    clock.tick()
    assert_equal(clock.t, 1*ms)
    assert_equal(clock._i, 1)
    clock._i = 5
    assert_equal(clock.t, 5*ms)

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
    clock._set_t_update_dt(t=0.1*ms)
    clock.dt = 0.1*ms
    clock._set_t_update_dt(t=0.1*ms)
    clock.dt = 0.3*ms
    assert_raises(ValueError, lambda: clock._set_t_update_dt(t=0.1*ms))
    clock.dt = 2*ms
    assert_raises(ValueError, lambda: clock._set_t_update_dt(t=0.1*ms))


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_defaultclock():        
    defaultclock.dt = 1*ms
    assert_equal(defaultclock.dt, 1*ms)
    assert defaultclock.name=='defaultclock'

if __name__=='__main__':
    test_clocks()
    restore_initial_state()
    test_clock_dt_change()
    restore_initial_state()
    test_defaultclock()
