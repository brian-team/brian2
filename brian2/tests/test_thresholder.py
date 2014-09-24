from numpy.testing import assert_equal

from brian2 import *

# We can only test C++ if weave is available
try:
    import scipy.weave
    codeobj_classes = [NumpyCodeObject, WeaveCodeObject]
except ImportError:
    # Can't test C++
    codeobj_classes = [NumpyCodeObject]


def test_simple_threshold():
    for codeobj_class in codeobj_classes:
        G = NeuronGroup(4, 'v : 1', threshold='v > 1',
                        codeobj_class=codeobj_class)
        G.v = [1.5, 0, 3, -1]
        s_mon = SpikeMonitor(G)
        net = Network(G, s_mon)
        net.run(defaultclock.dt)
        assert_equal(s_mon.count, np.array([1, 0, 1, 0]))

def test_scalar_threshold():
    c = 2
    for codeobj_class in codeobj_classes:
        G = NeuronGroup(4, '', threshold='c > 1',
                        codeobj_class=codeobj_class)
        s_mon = SpikeMonitor(G)
        net = Network(G, s_mon)
        net.run(defaultclock.dt)
        assert_equal(s_mon.count, np.array([1, 1, 1, 1]))


if __name__ == '__main__':
    test_simple_threshold()
    test_scalar_threshold()
