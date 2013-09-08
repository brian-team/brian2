import numpy as np
from numpy.testing.utils import assert_equal

from brian2 import *

# We can only test C++ if weave is availabe
try:
    import scipy.weave
    codeobj_classes = [NumpyCodeObject, WeaveCodeObject]
except ImportError:
    # Can't test C++
    codeobj_classes = [NumpyCodeObject]


def test_single_rates():
    for codeobj_class in codeobj_classes:
        # Specifying single rates
        P0 = PoissonGroup(10, 0*Hz, codeobj_class=codeobj_class)
        Pfull = PoissonGroup(10, 1. / defaultclock.dt,
                             codeobj_class=codeobj_class)

        # Basic properties
        assert len(P0) == len(Pfull) == 10
        assert len(repr(P0)) and len(str(P0))
        spikes_P0 = SpikeMonitor(P0)
        spikes_Pfull = SpikeMonitor(Pfull)
        net = Network(P0, Pfull, spikes_P0, spikes_Pfull)
        net.run(2*defaultclock.dt)
        assert_equal(spikes_P0.count, np.zeros(len(P0)))
        assert_equal(spikes_Pfull.count, 2 * np.ones(len(P0)))


def test_rate_arrays():
    for codeobj_class in codeobj_classes:
        P = PoissonGroup(2, np.array([0, 1./defaultclock.dt])*Hz,
                         codeobj_class=codeobj_class)
        spikes = SpikeMonitor(P)
        net = Network(P, spikes)
        net.run(2*defaultclock.dt)

        assert_equal(spikes.count, np.array([0, 2]))


def test_propagation():
    for codeobj_class in codeobj_classes:
        # Using a PoissonGroup as a source for Synapses should work as expected
        P = PoissonGroup(2, np.array([0, 1./defaultclock.dt])*Hz,
                         codeobj_class=codeobj_class)
        G = NeuronGroup(2, 'v:1')
        S = Synapses(P, G, pre='v+=1', connect='i==j')
        net = Network(P, S, G)
        net.run(0*ms)
        net.run(2*defaultclock.dt)

        assert_equal(G.v[:], np.array([0., 2.]))


if __name__ == '__main__':
    test_single_rates()
    test_rate_arrays()
    test_propagation()
