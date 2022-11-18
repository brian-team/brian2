import pytest
from numpy.testing import assert_equal

from brian2 import *
from brian2.devices.device import reinit_and_delete


@pytest.mark.standalone_compatible
def test_simple_threshold():
    G = NeuronGroup(4, "v : 1", threshold="v > 1")
    G.v = [1.5, 0, 3, -1]
    s_mon = SpikeMonitor(G)
    run(defaultclock.dt)
    assert_equal(s_mon.count, np.array([1, 0, 1, 0]))


@pytest.mark.standalone_compatible
def test_scalar_threshold():
    c = 2
    G = NeuronGroup(4, "", threshold="c > 1")
    s_mon = SpikeMonitor(G)
    run(defaultclock.dt)
    assert_equal(s_mon.count, np.array([1, 1, 1, 1]))


if __name__ == "__main__":
    test_simple_threshold()
    test_scalar_threshold()
