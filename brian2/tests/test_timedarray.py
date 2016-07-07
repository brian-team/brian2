from numpy.testing.utils import assert_equal, assert_raises
from nose import with_setup
from nose.plugins.attrib import attr

from brian2 import *
from brian2.devices.device import reinit_devices

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
    ta2d = TimedArray((np.linspace(0, 11, 12)*amp).reshape(4, 3), 1*ms)
    assert ta2d(-1*ms, 0) == 0*amp
    assert ta2d(0*ms, 0) == 0*amp
    assert ta2d(0*ms, 1) == 1*amp
    assert ta2d(1*ms, 1) == 4*amp
    assert_equal(ta2d(1*ms, [0, 1, 2]), [3, 4, 5]*amp)
    assert_equal(ta2d(15*ms, [0, 1, 2]), [9, 10, 11]*amp)


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_timedarray_semantics():
    # Make sure that timed arrays are interpreted as specifying the values
    # between t and t+dt (not between t-dt/2 and t+dt/2 as in Brian1)
    ta = TimedArray(array([0, 1]), dt=0.4*ms)
    G = NeuronGroup(1, 'value = ta(t) : 1', dt=0.1*ms)
    mon = StateMonitor(G, 'value', record=0)
    run(0.8*ms)
    assert_equal(mon[0].value, [0, 0, 0, 0, 1, 1, 1, 1])
    assert_equal(mon[0].value, ta(mon.t))

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_timedarray_no_units():
    ta = TimedArray(np.arange(10), dt=0.1*ms)
    G = NeuronGroup(1, 'value = ta(t) + 1: 1', dt=0.1*ms)
    mon = StateMonitor(G, 'value', record=True, dt=0.1*ms)
    run(1.1*ms)
    assert_equal(mon[0].value_, np.clip(np.arange(len(mon[0].t)), 0, 9) + 1)

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_timedarray_with_units():
    ta = TimedArray(np.arange(10)*amp, dt=0.1*ms)
    G = NeuronGroup(1, 'value = ta(t) + 2*nA: amp', dt=0.1*ms)
    mon = StateMonitor(G, 'value', record=True, dt=0.1*ms)
    run(1.1*ms)
    assert_equal(mon[0].value, np.clip(np.arange(len(mon[0].t)), 0, 9)*amp + 2*nA)

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_timedarray_2d():
    # 4 time steps, 3 neurons
    ta2d = TimedArray(np.arange(12).reshape(4, 3), dt=0.1*ms)
    G = NeuronGroup(3, 'value = ta2d(t, i) + 1: 1', dt=0.1*ms)
    mon = StateMonitor(G, 'value', record=True, dt=0.1*ms)
    run(0.5*ms)
    assert_equal(mon[0].value_, np.array([0, 3, 6, 9, 9]) + 1)
    assert_equal(mon[1].value_, np.array([1, 4, 7, 10, 10]) + 1)
    assert_equal(mon[2].value_, np.array([2, 5, 8, 11, 11]) + 1)


@attr('codegen-independent')
def test_timedarray_incorrect_use():
    ta = TimedArray(np.linspace(0, 10, 11), 1*ms)
    ta2d = TimedArray((np.linspace(0, 11, 12)*amp).reshape(4, 3), 1*ms)
    G = NeuronGroup(3, 'I : amp')
    # The weird formulation with the variable name is to get the variable into
    # the surrounding namespace of the setattr call
    assert_raises(ValueError, lambda: (ta2d, setattr(G, 'I', 'ta2d(t)*amp')))
    assert_raises(ValueError, lambda: (ta, setattr(G, 'I', 'ta(t, i)*amp')))
    assert_raises(ValueError, lambda: (ta, setattr(G, 'I', 'ta()*amp')))
    assert_raises(ValueError, lambda: (ta, setattr(G, 'I', 'ta*amp')))


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_timedarray_no_upsampling():
    # Test a TimedArray where no upsampling is necessary because the monitor's
    # dt is bigger than the TimedArray's
    ta = TimedArray(np.arange(10), dt=0.01*ms)
    G = NeuronGroup(1, 'value = ta(t): 1', dt=0.1*ms)
    mon = StateMonitor(G, 'value', record=True, dt=1*ms)
    run(2.1*ms)
    assert_equal(mon[0].value, [0, 9, 9])


#@attr('standalone-compatible')  # see FIXME comment below
@with_setup(teardown=reinit_devices)
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
    # FIXME: setting the time like this does not work for standalone
    net.t_ = float(16384*second - 5*ms)
    net.run(10*ms)
    assert_equal(mon[0].value[mon.t < 16384*second], 16383)
    assert_equal(mon[0].value[mon.t >= 16384*second], 16384)


if __name__ == '__main__':
    test_timedarray_direct_use()
    test_timedarray_semantics()
    test_timedarray_no_units()
    test_timedarray_with_units()
    test_timedarray_2d()
    test_timedarray_incorrect_use()
    test_timedarray_no_upsampling()
    test_long_timedarray()
