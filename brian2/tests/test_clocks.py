from brian2 import *
from numpy.testing import assert_raises, assert_equal
from nose import with_setup

@with_setup(teardown=restore_initial_state)
def test_clocks():
    clock = Clock(dt=1*ms)
    assert_equal(clock.t, 0*second)
    clock.tick()
    assert_equal(clock.t, 1*ms)
    assert_equal(clock.i, 1)
    clock.i = 5
    assert_equal(clock.t, 5*ms)
    clock.t = 10*ms
    assert_equal(clock.i, 10)
    clock.t_end = 100*ms
    assert_equal(clock.i_end, 100)
    clock.i_end = 200
    assert_equal(clock.t_end, 200*ms)
    clock.t_ = float(8*ms)
    assert_equal(clock.i, 8)
    clock.t = 0*ms
    clock.set_interval(0*ms, 10*ms)
    assert_equal(clock.running, True)
    clock.t = 9.9*ms
    assert_equal(clock.running, True)
    clock.t = 10*ms
    assert_equal(clock.running, False)
    clock.reinit()
    assert_equal(clock.t, 0*ms)

    clock = Clock()
    assert_equal(clock.dt, 0.1*ms)
    clock.dt = 1*ms
    assert_equal(clock.dt, 1*ms)

@with_setup(teardown=restore_initial_state)
def test_defaultclock():        
    defaultclock.dt = 1*ms
    assert_equal(defaultclock.dt, 1*ms)
    assert defaultclock.name=='defaultclock'

if __name__=='__main__':
    test_clocks()
    restore_initial_state()
    test_defaultclock()
