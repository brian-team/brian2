from nose import with_setup
from nose.plugins.attrib import attr
from numpy.testing import assert_equal

from brian2 import *
from brian2.devices.device import reinit_devices

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_simple_threshold():
    G = NeuronGroup(4, 'v : 1', threshold='v > 1')
    G.v = [1.5, 0, 3, -1]
    s_mon = SpikeMonitor(G)
    run(defaultclock.dt)
    assert_equal(s_mon.count, np.array([1, 0, 1, 0]))

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_scalar_threshold():
    c = 2
    G = NeuronGroup(4, '', threshold='c > 1')
    s_mon = SpikeMonitor(G)
    run(defaultclock.dt)
    assert_equal(s_mon.count, np.array([1, 1, 1, 1]))


if __name__ == '__main__':
    test_simple_threshold()
    test_scalar_threshold()
