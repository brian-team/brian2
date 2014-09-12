from brian2 import *
from numpy.testing import assert_raises, assert_equal
from nose import with_setup

@with_setup(teardown=restore_initial_state)
def test_clocks():
    clock = Clock(dt=1*ms)
    assert_equal(clock.t, 0*second)
    clock.tick()
    assert_equal(clock.t, 1*ms)
    assert_equal(clock._i, 1)
    clock._i = 5
    assert_equal(clock.t, 5*ms)


if __name__=='__main__':
    test_clocks()
    restore_initial_state()
