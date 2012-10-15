from brian2 import *
from brian2.memory.allocation import allocate_array
from numpy.testing import assert_raises, assert_equal
from nose import with_setup

@with_setup(teardown=restore_initial_state)
def test_allocate_array():
    arr = allocate_array(100)
    assert_equal(arr.shape, (100,))
    assert_equal(arr.dtype, float)
    arr = allocate_array((100, 2), dtype=int)
    assert_equal(arr.shape, (100, 2))
    assert_equal(arr.dtype, int)

if __name__=='__main__':
    test_allocate_array()
    