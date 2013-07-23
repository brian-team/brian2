import numpy as np
from numpy.testing.utils import assert_allclose, assert_equal

from brian2 import *

# We can only test C++ if weave is availabe
try:
    import scipy.weave
    languages = ['numpy', 'weave']
except ImportError:
    # Can't test C++
    languages = ['numpy']


def test_spike_monitor():
    language_before = brian_prefs.codegen.target
    for language in languages:
        brian_prefs.codegen.target = language
        defaultclock.t = 0*second
        G = NeuronGroup(2, '''dv/dt = rate : 1
                              rate: Hz''', threshold='v>1', reset='v=0')
        # We don't use 100 and 1000Hz, because then the membrane potential would
        # be exactly at 1 after 10 resp. 100 timesteps. Due to floating point
        # issues this will not be exact,
        G.rate = [101, 1001] * Hz

        mon = SpikeMonitor(G)
        net = Network(G, mon)
        net.run(10*ms)

        assert_allclose(mon.t[mon.i == 0], [9.9]*ms)
        assert_allclose(mon.t[mon.i == 1], np.arange(10)*ms + 0.9*ms)
        assert_equal(mon.count, np.array([1, 10]))

    brian_prefs.codegen.target = language_before


def test_state_monitor():
    # Record all neurons
    language_before = brian_prefs.codegen.target
    for language in languages:
        brian_prefs.codegen.target = language
        defaultclock.t = 0*second
        # Check that all kinds of variables can be recorded
        G = NeuronGroup(2, '''dv/dt = -v / (10*ms) : 1
                              f = clip(v, 0.1, 0.9) : 1
                              rate: Hz''', threshold='v>1', reset='v=0')
        G.rate = [100, 1000] * Hz
        G.v = 1

        # Use a single StateMonitor
        v_mon = StateMonitor(G, 'v', record=True)
        v_mon1 = StateMonitor(G, 'v', record=[1])

        # Use a StateMonitor for specified variables
        multi_mon = StateMonitor(G, ['v', 'f', 'rate'], record=True)
        multi_mon1 = StateMonitor(G, ['v', 'f', 'rate'], record=[1])

        net = Network(G, v_mon, v_mon1,
                      multi_mon, multi_mon1)
        net.run(10*ms)

        # Check v recording
        assert_allclose(v_mon.v,
                        np.exp(np.tile(-v_mon.t - defaultclock.dt, (2, 1)).T / (10*ms)))
        assert_allclose(v_mon.v_,
                        np.exp(np.tile(-v_mon.t_ - defaultclock.dt_, (2, 1)).T / float(10*ms)))
        assert_equal(v_mon.v, multi_mon.v)
        assert_equal(v_mon.v_, multi_mon.v_)
        assert_equal(v_mon.v[:, 1:2], v_mon1.v)
        assert_equal(multi_mon.v[:, 1:2], multi_mon1.v)

        # Other variables
        assert_equal(multi_mon.rate_, np.tile(np.atleast_2d(G.rate_),
                                             (multi_mon.rate.shape[0], 1)))
        assert_equal(multi_mon.rate[:, 1:2], multi_mon1.rate)
        assert_allclose(np.clip(multi_mon.v, 0.1, 0.9), multi_mon.f)
        assert_allclose(np.clip(multi_mon1.v, 0.1, 0.9), multi_mon1.f)

    brian_prefs.codegen.target = language_before

if __name__ == '__main__':
    test_spike_monitor()
    test_state_monitor()

