import numpy as np
from numpy.testing.utils import assert_equal
from nose.plugins.attrib import attr

from brian2 import *


@attr('codegen-independent')
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


def test_timedarray_semantics():
    # Make sure that timed arrays are interpreted as specifying the values
    # between t and t+dt (not between t-dt/2 and t+dt/2 as in Brian1)
    ta = TimedArray(array([0, 1]), dt=0.4*ms)
    G = NeuronGroup(1, 'value = ta(t) : 1', dt=0.1*ms)
    mon = StateMonitor(G, 'value', record=0)
    net = Network(G, mon)
    net.run(0.8*ms)
    assert_equal(mon[0].value, [0, 0, 0, 0, 1, 1, 1, 1])
    assert_equal(mon[0].value, ta(mon.t))


def test_timedarray_no_units():
    ta = TimedArray(np.arange(10), dt=0.1*ms)
    G = NeuronGroup(1, 'value = ta(t) + 1: 1', dt=0.1*ms)
    mon = StateMonitor(G, 'value', record=True, dt=0.1*ms)
    net = Network(G, mon)
    net.run(1.1*ms)
    assert_equal(mon[0].value_, np.clip(np.arange(len(mon[0].t)), 0, 9) + 1)


def test_timedarray_with_units():
    ta = TimedArray(np.arange(10)*amp, dt=0.1*ms)
    G = NeuronGroup(1, 'value = ta(t) + 2*nA: amp', dt=0.1*ms)
    mon = StateMonitor(G, 'value', record=True, dt=0.1*ms)
    net = Network(G, mon)
    net.run(1.1*ms)
    assert_equal(mon[0].value, np.clip(np.arange(len(mon[0].t)), 0, 9)*amp + 2*nA)


def test_timedarray_no_upsampling():
    # Test a TimedArray where no upsampling is necessary because the monitor's
    # dt is bigger than the TimedArray's
    ta = TimedArray(np.arange(10), dt=0.01*ms)
    G = NeuronGroup(1, 'value = ta(t): 1', dt=0.1*ms)
    mon = StateMonitor(G, 'value', record=True, dt=1*ms)
    net = Network(G, mon)
    net.run(2.1*ms)
    assert_equal(mon[0].value, [0, 9, 9])


def test_long_timedarray():
    '''
    Use a very long timedarray (with a big dt), where the upsampling can lead
    to integer overflow.
    '''
    ta = TimedArray(np.arange(16385), dt=1*second)
    G = NeuronGroup(1, 'value = ta(t) : 1')
    mon = StateMonitor(G, 'value', record=True)
    net = Network(G, mon)
    # We'll start the simulation close to the critical boundary
    net.t_ = float(16384*second - 5*ms)
    net.run(10*ms)
    assert all(mon[0].value[mon.t < 16384*second] == 16383)
    assert all(mon[0].value[mon.t >= 16384*second] == 16384)


if __name__ == '__main__':
    test_timedarray_direct_use()
    test_timedarray_semantics()
    test_timedarray_no_units()
    test_timedarray_with_units()
    test_timedarray_no_upsampling()
    test_long_timedarray()