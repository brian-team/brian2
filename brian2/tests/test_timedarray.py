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


def test_timedarray_direct_use():
    ta = TimedArray(np.linspace(0, 10, 11), 1*ms)
    assert ta(-1*ms) == 0
    assert ta(5*ms) == 5
    assert ta(10*ms) == 10
    assert ta(15*ms) == 10
    ta = TimedArray(np.linspace(0, 10, 11)*amp, 1*ms)
    assert ta(-1*ms) == 0*amp
    assert ta(5*ms) == 5*amp
    assert ta(10*ms) == 10*amp
    assert ta(15*ms) == 10*amp


def test_timedarray_no_units():
    ta = TimedArray(np.linspace(0, 10, 11), .9*ms)
    for codeobj_class in codeobj_classes:
        G = NeuronGroup(1, 'value = ta(t) : 1', codeobj_class=codeobj_class)
        mon = StateMonitor(G, 'value', record=True)
        net = Network(G, mon)
        net.run(11*ms)
        assert_equal(mon[0].value_,
                     np.clip(np.int_(mon[0].t / (.9*ms) + 0.5), 0, 10))


def test_timedarray_with_units():
    ta = TimedArray(np.linspace(0, 10, 11)*amp, .9*ms)
    for codeobj_class in codeobj_classes:
        G = NeuronGroup(1, 'value = ta(t) : amp', codeobj_class=codeobj_class)
        mon = StateMonitor(G, 'value', record=True)
        net = Network(G, mon)
        net.run(11*ms)
        assert_equal(mon[0].value,
                     np.clip(np.int_(mon[0].t / (.9*ms) + 0.5), 0, 10)*amp)

if __name__ == '__main__':
    test_timedarray_direct_use()
    test_timedarray_no_units()
    test_timedarray_with_units()