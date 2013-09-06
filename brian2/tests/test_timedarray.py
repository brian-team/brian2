import numpy as np

from brian2 import *

def test_timedarray_direct_use():
    ta = TimedArray('ta', np.linspace(0, 10, 11), 1*ms)
    assert ta(-1*ms) == 0
    assert ta(5*ms) == 5
    assert ta(10*ms) == 10
    assert ta(15*ms) == 10
    ta = TimedArray('ta', np.linspace(0, 10, 11)*amp, 1*ms)
    assert ta(-1*ms) == 0*amp
    assert ta(5*ms) == 5*amp
    assert ta(10*ms) == 10*amp
    assert ta(15*ms) == 10*amp


def test_timedarray_no_units():
    ta = TimedArray('ta', np.linspace(0, 10, 11), 1*ms)
    G = NeuronGroup(1, 'value = ta(t) : 1')
    mon = StateMonitor(G, 'value', record=True)
    net = Network(G, mon)
    net.run(12*ms)
    print mon.value_


def test_timedarray_with_units():
    ta = TimedArray('ta', np.linspace(0, 10, 11)*amp, 1*ms)
    G = NeuronGroup(1, 'value = ta(t) : amp')
    mon = StateMonitor(G, 'value', record=True)
    net = Network(G, mon)
    net.run(12*ms)
    print mon.value

if __name__ == '__main__':
    test_timedarray_direct_use()
    test_timedarray_no_units()
    test_timedarray_with_units()