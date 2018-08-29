import weakref
import copy
import logging
import tempfile
import os
import uuid

import numpy as np
from numpy.testing import assert_equal, assert_raises
from nose import with_setup
from nose.plugins.attrib import attr

from brian2 import (Clock, Network, ms, us, second, BrianObject, defaultclock,
                    run, stop, NetworkOperation, network_operation,
                    restore_initial_state, MagicError, Synapses,
                    NeuronGroup, StateMonitor, SpikeMonitor,
                    SpikeGeneratorGroup,
                    PopulationRateMonitor, MagicNetwork, magic_network,
                    PoissonGroup, Hz, collect, store, restore, BrianLogger,
                    start_scope, prefs, profiling_summary, Quantity, TimedArray)
from brian2.core.network import schedule_propagation_offset, scheduling_summary
from brian2.devices.device import (reinit_devices, Device, all_devices,
                                   set_device, get_device, reset_device, device)
from brian2.utils.logger import catch_logs
from brian2.tests.utils import assert_allclose

@attr('codegen-independent')
def test_incorrect_network_use():
    '''Test some wrong uses of `Network` and `MagicNetwork`'''
    assert_raises(TypeError, lambda: Network(name='mynet',
                                             anotherkwd='does not exist'))
    assert_raises(TypeError, lambda: Network('not a BrianObject'))
    net = Network()
    assert_raises(TypeError, lambda: net.add('not a BrianObject'))
    assert_raises(ValueError, lambda: MagicNetwork())
    G = NeuronGroup(10, 'v:1')
    net.add(G)
    assert_raises(TypeError, lambda: net.remove(object()))
    assert_raises(MagicError, lambda: magic_network.add(G))
    assert_raises(MagicError, lambda: magic_network.remove(G))


@attr('codegen-independent')
def test_network_contains():
    '''
    Test `Network.__contains__`.
    '''
    G = NeuronGroup(1, 'v:1', name='mygroup')
    net = Network(G)
    assert 'mygroup' in net
    assert 'neurongroup' not in net


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_empty_network():
    # Check that an empty network functions correctly
    net = Network()
    net.run(1*second)

class Counter(BrianObject):
    add_to_magic_network = True
    def __init__(self, **kwds):
        super(Counter, self).__init__(**kwds)
        self.count = 0

    def run(self):
        self.count += 1


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_network_single_object():
    # Check that a network with a single object functions correctly
    x = Counter()
    net = Network(x)
    net.run(1*ms)
    assert_equal(x.count, 10)


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_network_two_objects():
    # Check that a network with two objects and the same clock function correctly
    x = Counter(order=5)
    y = Counter(order=6)
    net = Network()
    net.add([x, [y]]) # check that a funky way of adding objects work correctly
    net.run(1*ms)
    assert_equal(net.objects[0].order, 5)
    assert_equal(net.objects[1].order, 6)
    assert_equal(len(net.objects), 2)
    assert_equal(x.count, 10)
    assert_equal(y.count, 10)


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_network_from_dict():
    # Check that a network from a dictionary works
    x = Counter()
    y = Counter()
    d = dict(a=x, b=y)
    net = Network()
    net.add(d)
    net.run(1*ms)
    assert_equal(len(net.objects), 2)
    assert_equal(x.count, 10)
    assert_equal(y.count, 10)


class NameLister(BrianObject):
    add_to_magic_network = True
    updates = []

    def __init__(self, **kwds):
        super(NameLister, self).__init__(**kwds)

    def run(self):
        NameLister.updates.append(self.name)


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_network_different_clocks():
    NameLister.updates[:] = []
    # Check that a network with two different clocks functions correctly
    x = NameLister(name='x', dt=.1*ms, order=0)
    y = NameLister(name='y', dt=1*ms, order=1)
    net = Network(x, y)
    net.run(100*second+defaultclock.dt, report='text')
    updates = ''.join(NameLister.updates)[2:]  # ignore the first time step
    assert updates == ('xxxxxxxxxxy'*100000)


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_network_different_when():
    # Check that a network with different when attributes functions correctly
    NameLister.updates[:] = []
    x = NameLister(name='x', when='start')
    y = NameLister(name='y', when='end')
    net = Network(x, y)
    net.run(0.3*ms)
    assert_equal(''.join(NameLister.updates), 'xyxyxy')

@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_network_default_schedule():
    net = Network()
    assert net.schedule == ['start', 'groups', 'thresholds', 'synapses', 'resets', 'end']
    # Set the preference and check that the change is taken into account
    prefs.core.network.default_schedule = list(reversed(['start', 'groups', 'thresholds', 'synapses', 'resets', 'end']))
    assert net.schedule == list(reversed(['start', 'groups', 'thresholds', 'synapses', 'resets', 'end']))

@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_network_schedule_change():
    # Check that a changed schedule is taken into account correctly
    NameLister.updates[:] = []
    x = NameLister(name='x', when='thresholds')
    y = NameLister(name='y', when='resets')
    net = Network(x, y)
    net.run(0.3*ms)
    assert_equal(''.join(NameLister.updates), 'xyxyxy')
    NameLister.updates[:] = []
    net.schedule = ['start', 'groups', 'synapses', 'resets', 'thresholds', 'end']
    net.run(0.3*ms)
    assert_equal(''.join(NameLister.updates), 'yxyxyx')

@attr('codegen-independent')
def test_network_before_after_schedule():
    # Test that before... and after... slot names can be used
    NameLister.updates[:] = []
    x = NameLister(name='x', when='before_resets')
    y = NameLister(name='y', when='after_thresholds')
    net = Network(x, y)
    net.schedule = ['thresholds', 'resets', 'end']
    net.run(0.3*ms)
    assert_equal(''.join(NameLister.updates), 'yxyxyx')

@attr('codegen-independent')
def test_network_custom_slots():
    # Check that custom slots can be inserted into the schedule
    NameLister.updates[:] = []
    x = NameLister(name='x', when='thresholds')
    y = NameLister(name='y', when='in_between')
    z = NameLister(name='z', when='resets')
    net = Network(x, y, z)
    net.schedule = ['start', 'groups', 'thresholds', 'in_between', 'synapses', 'resets', 'end']
    net.run(0.3*ms)
    assert_equal(''.join(NameLister.updates), 'xyzxyzxyz')

@attr('codegen-independent')
def test_network_incorrect_schedule():
    # Test that incorrect arguments provided to schedule raise errors
    net = Network()
    # net.schedule = object()
    assert_raises(TypeError, setattr, net, 'schedule', object())
    # net.schedule = 1
    assert_raises(TypeError, setattr, net, 'schedule', 1)
    # net.schedule = {'slot1', 'slot2'}
    assert_raises(TypeError, setattr, net, 'schedule', {'slot1', 'slot2'})
    # net.schedule = ['slot', 1]
    assert_raises(TypeError, setattr, net, 'schedule', ['slot', 1])
    # net.schedule = ['start', 'after_start']
    assert_raises(ValueError, setattr, net, 'schedule', ['start', 'after_start'])
    # net.schedule = ['before_start', 'start']
    assert_raises(ValueError, setattr, net, 'schedule', ['before_start', 'start'])

@attr('codegen-independent')
@with_setup(teardown=reinit_devices)
def test_schedule_warning():
    previous_device = get_device()
    from uuid import uuid4
    # TestDevice1 supports arbitrary schedules, TestDevice2 does not
    class TestDevice1(Device):
        # These functions are needed during the setup of the defaultclock
        def get_value(self, var):
            return np.array([0.0001])
        def add_array(self, var):
            pass
        def init_with_zeros(self, var, dtype):
            pass
        def fill_with_array(self, var, arr):
            pass
    class TestDevice2(TestDevice1):
        def __init__(self):
            super(TestDevice2, self).__init__()
            self.network_schedule = ['start', 'groups', 'synapses',
                                     'thresholds', 'resets', 'end']

    # Unique names are important for getting the warnings again for multiple
    # runs of the test suite
    name1 = 'testdevice_' + str(uuid4())
    name2 = 'testdevice_' + str(uuid4())
    all_devices[name1] = TestDevice1()
    all_devices[name2] = TestDevice2()

    set_device(name1)
    assert schedule_propagation_offset() == 0*ms
    net = Network()
    assert schedule_propagation_offset(net) == 0*ms

    # Any schedule should work
    net.schedule = list(reversed(net.schedule))
    with catch_logs() as l:
        net.run(0*ms)
        assert len(l) == 0, 'did not expect a warning'

    assert schedule_propagation_offset(net) == defaultclock.dt

    set_device(name2)
    assert schedule_propagation_offset() == defaultclock.dt

    # Using the correct schedule should work
    net.schedule = ['start', 'groups', 'synapses', 'thresholds', 'resets', 'end']
    with catch_logs() as l:
        net.run(0*ms)
        assert len(l) == 0, 'did not expect a warning'
    assert schedule_propagation_offset(net) == defaultclock.dt

    # Using another (e.g. the default) schedule should raise a warning
    net.schedule = None
    with catch_logs() as l:
        net.run(0*ms)
        assert len(l) == 1 and l[0][1].endswith('schedule_conflict')
    reset_device(previous_device)


@attr('codegen-independent')
def test_scheduling_summary_magic():
    basename = 'name' + str(uuid.uuid4()).replace('-', '_')
    group = NeuronGroup(10, 'dv/dt = -v/(10*ms) : 1', threshold='v>1',
                        reset='v=1', name=basename)
    group.run_regularly('v = rand()', dt=defaultclock.dt*10, when='end')
    state_mon = StateMonitor(group, 'v', record=True, name=basename+'_sm')
    inactive_state_mon = StateMonitor(group, 'v', record=True,
                                      name=basename+'_sm_ia', when='after_end')
    inactive_state_mon.active = False
    summary_before = scheduling_summary()
    assert [entry.name for entry in summary_before.entries] == [basename+'_sm',
                                                                basename+'_stateupdater',
                                                                basename+'_thresholder',
                                                                basename+'_resetter',
                                                                basename+'_run_regularly',
                                                                basename+'_sm_ia']
    assert [entry.when for entry in summary_before.entries] == ['start',
                                                                'groups',
                                                                'thresholds',
                                                                'resets',
                                                                'end',
                                                                'after_end']
    assert [entry.dt for entry in summary_before.entries] == [defaultclock.dt,
                                                              defaultclock.dt,
                                                              defaultclock.dt,
                                                              defaultclock.dt,
                                                              defaultclock.dt*10,
                                                              defaultclock.dt]
    assert [entry.active for entry in summary_before.entries] == [True,
                                                                  True,
                                                                  True,
                                                                  True,
                                                                  True,
                                                                  False]
    assert len(str(summary_before))
    assert len(summary_before._repr_html_())
    run(defaultclock.dt)
    summary_after = scheduling_summary()
    assert str(summary_after) == str(summary_before)
    assert summary_after._repr_html_() == summary_before._repr_html_()


@attr('codegen-independent')
def test_scheduling_summary():
    basename = 'name' + str(uuid.uuid4()).replace('-', '_')
    group = NeuronGroup(10, 'dv/dt = -v/(10*ms) : 1', threshold='v>1',
                        reset='v=1', name=basename)
    group.run_regularly('v = rand()', dt=defaultclock.dt * 10, when='end')
    state_mon = StateMonitor(group, 'v', record=True, name=basename + '_sm')
    inactive_state_mon = StateMonitor(group, 'v', record=True,
                                      name=basename + '_sm_ia',
                                      when='after_end')
    inactive_state_mon.active = False
    net = Network(group, state_mon, inactive_state_mon)
    summary_before = scheduling_summary(net)
    assert [entry.name for entry in summary_before.entries] == [basename+'_sm',
                                                                basename+'_stateupdater',
                                                                basename+'_thresholder',
                                                                basename+'_resetter',
                                                                basename+'_run_regularly',
                                                                basename+'_sm_ia']
    assert [entry.when for entry in summary_before.entries] == ['start',
                                                                'groups',
                                                                'thresholds',
                                                                'resets',
                                                                'end',
                                                                'after_end']
    assert [entry.dt for entry in summary_before.entries] == [defaultclock.dt,
                                                              defaultclock.dt,
                                                              defaultclock.dt,
                                                              defaultclock.dt,
                                                              defaultclock.dt*10,
                                                              defaultclock.dt]
    assert [entry.active for entry in summary_before.entries] == [True,
                                                                  True,
                                                                  True,
                                                                  True,
                                                                  True,
                                                                  False]
    assert len(str(summary_before))
    assert len(summary_before._repr_html_())
    run(defaultclock.dt)
    summary_after = scheduling_summary(net)
    assert str(summary_after) == str(summary_before)
    assert summary_after._repr_html_() == summary_before._repr_html_()


class Preparer(BrianObject):
    add_to_magic_network = True
    def __init__(self, **kwds):
        super(Preparer, self).__init__(**kwds)
        self.did_reinit = False
        self.did_pre_run = False
        self.did_post_run = False
    def reinit(self, level=0):
        self.did_reinit = True
    def before_run(self, namespace=None, level=0):
        self.did_pre_run = True
    def after_run(self):
        self.did_post_run = True        


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_magic_network():
    # test that magic network functions correctly
    x = Counter()
    y = Counter()
    run(10*ms)
    assert_equal(x.count, 100)
    assert_equal(y.count, 100)

    assert len(repr(magic_network))  # very basic test...
    assert len(str(magic_network))  # very basic test...

class Stopper(BrianObject):
    add_to_magic_network = True
    def __init__(self, stoptime, stopfunc, **kwds):
        super(Stopper, self).__init__(**kwds)
        self.stoptime = stoptime
        self.stopfunc = stopfunc

    def run(self):
        self.stoptime -= 1
        if self.stoptime<=0:
            self.stopfunc()

@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_network_stop():
    # test that Network.stop and global stop() work correctly
    net = Network()
    x = Stopper(10, net.stop)
    net.add(x)
    net.run(10*ms)
    assert_equal(defaultclock.t, 1*ms)
    
    x = Stopper(10, stop)
    net = Network(x)
    net.run(10*ms)
    assert_equal(defaultclock.t, 1*ms)


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_network_operations():
    # test NetworkOperation and network_operation
    seq = []
    def f1():
        seq.append('a')
    op1 = NetworkOperation(f1, when='start', order=1)
    @network_operation
    def f2():
        seq.append('b')
    @network_operation(when='end', order=1)
    def f3():
        seq.append('c')

    # In complex frameworks, network operations might be object methods that
    # access some common data
    class Container(object):
        def __init__(self):
            self.g1_data = 'B'
            self.g2_data = 'C'

        def g1(self):
            seq.append(self.g1_data)

        def g2(self):
            seq.append(self.g2_data)

    c = Container()
    c_op1 = NetworkOperation(c.g1)
    c_op2 = NetworkOperation(c.g2, when='end', order=1)
    net = Network(op1, f2, f3, c_op1, c_op2)
    net.run(1*ms)

    assert_equal(''.join(seq), 'bBacC'*10)

@attr('codegen-independent')
def test_incorrect_network_operations():
    # Network operations with more than one argument are not allowed
    def func(x, y):
        pass

    class Container(object):
        def func(self, x, y):
            pass
    c = Container()

    assert_raises(TypeError, lambda: NetworkOperation(func))
    assert_raises(TypeError, lambda: NetworkOperation(c.func))

    # Incorrect use of @network_operation -- it does not work on an instance
    # method
    try:
        class Container(object):
            @network_operation
            def func(self):
                pass
        raise AssertionError('expected a TypeError')
    except TypeError:
        pass  # this is what we expected

@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_network_active_flag():
    # test that the BrianObject.active flag is recognised by Network.run
    x = Counter()
    y = Counter()
    y.active = False
    run(1*ms)
    assert_equal(x.count, 10)
    assert_equal(y.count, 0)


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_network_t():
    # test that Network.t works as expected
    x = Counter(dt=1*ms)
    y = Counter(dt=2*ms)
    net = Network(x, y)
    net.run(4*ms)
    assert_equal(net.t, 4*ms)
    net.run(1*ms)
    assert_equal(net.t, 5*ms)
    assert_equal(x.count, 5)
    assert_equal(y.count, 3)
    net.run(0.5*ms) # should only update x
    assert_equal(net.t, 5.5*ms)
    assert_equal(x.count, 6)
    assert_equal(y.count, 3)
    net.run(0.5*ms) # shouldn't do anything
    assert_equal(net.t, 6*ms)
    assert_equal(x.count, 6)
    assert_equal(y.count, 3)
    net.run(0.5*ms) # should update x and y
    assert_equal(net.t, 6.5*ms)
    assert_equal(x.count, 7)
    assert_equal(y.count, 4)
    
    del x, y, net

    # now test with magic run
    x = Counter(dt=1*ms)
    y = Counter(dt=2*ms)
    run(4*ms)
    assert_equal(magic_network.t, 4*ms)
    assert_equal(x.count, 4)
    assert_equal(y.count, 2)
    run(4*ms)
    assert_equal(magic_network.t, 8*ms)
    assert_equal(x.count, 8)
    assert_equal(y.count, 4)
    run(1*ms)
    assert_equal(magic_network.t, 9*ms)
    assert_equal(x.count, 9)
    assert_equal(y.count, 5)


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_incorrect_dt_defaultclock():
    defaultclock.dt = 0.5*ms
    G = NeuronGroup(1, 'dv/dt = -v / (10*ms) : 1')
    net = Network(G)
    net.run(0.5*ms)
    defaultclock.dt = 1*ms
    assert_raises(ValueError, lambda: net.run(0*ms))


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_incorrect_dt_custom_clock():
    clock = Clock(dt=0.5*ms)
    G = NeuronGroup(1, 'dv/dt = -v / (10*ms) : 1', clock=clock)
    net = Network(G)
    net.run(0.5*ms)
    clock.dt = 1*ms
    assert_raises(ValueError, lambda: net.run(0*ms))


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_network_remove():
    x = Counter()
    y = Counter()
    net = Network(x, y)
    net.remove(y)
    net.run(1*ms)
    assert_equal(x.count, 10)
    assert_equal(y.count, 0)
    # the relevance of this test is when we use weakref.proxy objects in
    # Network.objects, we should be able to add and remove these from
    # the Network just as much as the original objects
    # TODO: Does this test make sense now that Network does not store weak
    #       references by default?
    for obj in copy.copy(net.objects):
        net.remove(obj)
    net.run(1*ms)
    assert_equal(x.count, 10)
    assert_equal(y.count, 0)

class NoninvalidatingCounter(Counter):
    add_to_magic_network = True
    invalidates_magic_network = False

@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_invalid_magic_network():
    x = Counter()
    run(1*ms)
    assert_equal(x.count, 10)
    y = Counter()
    try:
        run(1*ms)
        raise AssertionError('Expected a MagicError')
    except MagicError:
        pass  # this is expected
    del x, y
    x = Counter()
    run(1*ms)
    y = NoninvalidatingCounter()
    run(1*ms)
    assert_equal(x.count, 20)
    assert_equal(y.count, 10)
    del y
    run(1*ms)
    assert_equal(x.count, 30)
    del x
    x = Counter()
    run(1*ms)
    assert_equal(magic_network.t, 1*ms)
    del x
    x = Counter()
    y = Counter()
    run(1*ms)
    assert_equal(x.count, 10)
    assert_equal(y.count, 10) 


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_multiple_networks_invalid():
    x = Counter()
    net = Network(x)
    net.run(1*ms)
    try:
        run(1*ms)
        raise AssertionError('Expected a RuntimeError')
    except RuntimeError:
        pass  # this is expected

    try:
        net2 = Network(x)
        raise AssertionError('Expected a RuntimeError')
    except RuntimeError:
        pass  # this is expected


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_magic_weak_reference():
    '''
    Test that holding a weak reference to an object does not make it get
    simulated.'''

    G1 = NeuronGroup(1, 'v:1')

    # this object should not be included
    G2 = weakref.ref(NeuronGroup(1, 'v:1'))

    with catch_logs(log_level=logging.DEBUG) as l:
        run(1*ms)
        # Check the debug messages for the number of included objects
        magic_objects = [msg[2] for msg in l
                         if msg[1] == 'brian2.core.magic.magic_objects'][0]
        assert '2 objects' in magic_objects, 'Unexpected log message: %s' % magic_objects


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_magic_unused_object():
    '''Test that creating unused objects does not affect the magic system.'''
    def create_group():
        # Produce two objects but return only one
        G1 = NeuronGroup(1, 'v:1')  # no Thresholder or Resetter
        G2 = NeuronGroup(1, 'v:1') # This object should be garbage collected
        return G1

    G = create_group()
    with catch_logs(log_level=logging.DEBUG) as l:
        run(1*ms)

        # Check the debug messages for the number of included objects
        magic_objects = [msg[2] for msg in l
                         if msg[1] == 'brian2.core.magic.magic_objects'][0]
        assert '2 objects' in magic_objects, 'Unexpected log message: %s' % magic_objects


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_network_access():
    x = Counter(name='counter')
    net = Network(x)
    assert len(net) == 1
    assert len(repr(net))  # very basic test...
    assert len(str(net))  # very basic test...

    # accessing objects
    assert net['counter'] is x
    assert_raises(TypeError, lambda: net[123])
    assert_raises(TypeError, lambda: net[1:3])
    assert_raises(KeyError, lambda: net['non-existing'])

    objects = [obj for obj in net]
    assert set(objects) == set(net.objects)

    # deleting objects
    del net['counter']
    assert_raises(TypeError, lambda: net.__delitem__(123))
    assert_raises(TypeError, lambda: net.__delitem__(slice(1, 3)))
    assert_raises(KeyError, lambda: net.__delitem__('counter'))


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_dependency_check():
    def create_net():
        G = NeuronGroup(10, 'v: 1', threshold='False')
        dependent_objects = [
                             StateMonitor(G, 'v', record=True),
                             SpikeMonitor(G),
                             PopulationRateMonitor(G),
                             Synapses(G, G, on_pre='v+=1')
                             ]
        return dependent_objects

    dependent_objects = create_net()
    # Trying to simulate the monitors/synapses without the group should fail
    for obj in dependent_objects:
        assert_raises(ValueError, lambda: Network(obj).run(0*ms))

    # simulation with a magic network should work when we have an explicit
    # reference to one of the objects, but the object should be inactive and
    # we should get a warning
    assert all(obj.active for obj in dependent_objects)
    for obj in dependent_objects:  # obj is our explicit reference
        with catch_logs() as l:
            run(0*ms)
            dependency_warnings = [msg[2] for msg in l
                                   if msg[1] == 'brian2.core.magic.dependency_warning']
            assert len(dependency_warnings) == 1
        assert not obj.active


@with_setup(teardown=restore_initial_state)
def test_loop():
    '''
    Somewhat realistic test with a loop of magic networks
    '''
    def run_simulation():
        G = NeuronGroup(10, 'dv/dt = -v / (10*ms) : 1',
                        reset='v=0', threshold='v>1')
        G.v = np.linspace(0, 1, 10)
        run(1*ms)
        # We return potentially problematic references to a VariableView
        return G.v

    # First run
    with catch_logs(log_level=logging.DEBUG) as l:
        v = run_simulation()
        assert v[0] == 0 and 0 < v[-1] < 1
        # Check the debug messages for the number of included objects
        magic_objects = [msg[2] for msg in l
                         if msg[1] == 'brian2.core.magic.magic_objects'][0]
        assert '4 objects' in magic_objects


    # Second run
    with catch_logs(log_level=logging.DEBUG) as l:
        v = run_simulation()
        assert v[0] == 0 and 0 < v[-1] < 1
        # Check the debug messages for the number of included objects
        magic_objects = [msg[2] for msg in l
                         if msg[1] == 'brian2.core.magic.magic_objects'][0]
        assert '4 objects' in magic_objects


@attr('codegen-independent')
def test_magic_collect():
    '''
    Make sure all expected objects are collected in a magic network
    '''
    P = PoissonGroup(10, rates=100*Hz)
    G = NeuronGroup(10, 'v:1', threshold='False')
    S = Synapses(G, G, '')

    state_mon = StateMonitor(G, 'v', record=True)
    spike_mon = SpikeMonitor(G)
    rate_mon = PopulationRateMonitor(G)

    objects = collect()

    assert len(objects) == 6, ('expected %d objects, got %d' % (6, len(objects)))

from contextlib import contextmanager
from StringIO import StringIO
import sys

@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


@attr('codegen-independent')
def test_progress_report():
    '''
    Very basic test of progress reporting
    '''
    G = NeuronGroup(1, '')
    net = Network(G)

    # No output
    with captured_output() as (out, err):
        net.run(1*ms, report=None)
    # There should be at least two lines of output
    out, err = out.getvalue(), err.getvalue()
    assert len(out) == 0 and len(err) == 0

    with captured_output() as (out, err):
        net.run(1*ms)
    # There should be at least two lines of output
    out, err = out.getvalue(), err.getvalue()
    assert len(out) == 0 and len(err) == 0

    # Progress should go to stdout
    with captured_output() as (out, err):
        net.run(1*ms, report='text')
    # There should be at least two lines of output
    out, err = out.getvalue(), err.getvalue()
    assert len(out.split('\n')) >= 2 and len(err) == 0

    with captured_output() as (out, err):
        net.run(1*ms, report='stdout')
    # There should be at least two lines of output
    out, err = out.getvalue(), err.getvalue()
    assert len(out.split('\n')) >= 2 and len(err) == 0

    # Progress should go to stderr
    with captured_output() as (out, err):
        net.run(1*ms, report='stderr')
    # There should be at least two lines of output
    out, err = out.getvalue(), err.getvalue()
    assert len(err.split('\n')) >= 2 and len(out) == 0

    # Custom function
    calls = []
    def capture_progress(elapsed, complete, start, duration):
        calls.append((elapsed, complete, start, duration))
    with captured_output() as (out, err):
        net.run(1*ms, report=capture_progress)
    out, err = out.getvalue(), err.getvalue()

    assert len(err) == 0 and len(out) == 0
    # There should be at least a call for the start and the end
    assert len(calls) >= 2 and calls[0][1] == 0.0 and calls[-1][1] == 1.0


@attr('codegen-independent')
def test_progress_report_incorrect():
    '''
    Test wrong use of the report option
    '''
    G = NeuronGroup(1, '')
    net = Network(G)
    assert_raises(ValueError, lambda: net.run(1*ms, report='unknown'))
    assert_raises(TypeError, lambda: net.run(1*ms, report=object()))


@attr('standalone-compatible', 'multiple-runs')
@with_setup(teardown=reinit_devices)
def test_multiple_runs_report_standalone():
    group = NeuronGroup(1, 'dv/dt = 1*Hz : 1')
    run(1*ms, report='text')
    run(1*ms)
    device.build(direct_call=False, **device.build_options)


@attr('standalone-compatible', 'multiple-runs')
@with_setup(teardown=reinit_devices)
def test_multiple_runs_report_standalone_2():
    group = NeuronGroup(1, 'dv/dt = 1*Hz : 1')
    run(1*ms)
    run(1*ms, report='text')
    device.build(direct_call=False, **device.build_options)


@attr('standalone-compatible', 'multiple-runs')
@with_setup(teardown=reinit_devices)
def test_multiple_runs_report_standalone_3():
    group = NeuronGroup(1, 'dv/dt = 1*Hz : 1')
    run(1*ms, report='text')
    run(1*ms, report='text')
    device.build(direct_call=False, **device.build_options)


# This tests a specific limitation of the C++ standalone mode (cannot mix
# multiple report methods)
@attr('cpp_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_multiple_runs_report_standalone_incorrect():
    set_device('cpp_standalone', build_on_run=False)
    group = NeuronGroup(1, 'dv/dt = 1*Hz : 1')
    run(1*ms, report='text')
    assert_raises(NotImplementedError, lambda: run(1*ms, report='stderr'))


@attr('codegen-independent')
def test_store_restore():
    source = NeuronGroup(10, '''dv/dt = rates : 1
                                rates : Hz''', threshold='v>1', reset='v=0')
    source.rates = 'i*100*Hz'
    target = NeuronGroup(10, 'v:1')
    synapses = Synapses(source, target, model='w:1', on_pre='v+=w')
    synapses.connect(j='i')
    synapses.w = 'i*1.0'
    synapses.delay = 'i*ms'
    state_mon = StateMonitor(target, 'v', record=True)
    spike_mon = SpikeMonitor(source)
    net = Network(source, target, synapses, state_mon, spike_mon)
    net.store()  # default time slot
    net.run(10*ms)
    net.store('second')
    net.run(10*ms)
    v_values = state_mon.v[:, :]
    spike_indices, spike_times = spike_mon.it_
    net.restore() # Go back to beginning
    assert defaultclock.t == 0*ms
    assert net.t == 0*ms
    net.run(20*ms)
    assert_equal(v_values, state_mon.v[:, :])
    assert_equal(spike_indices, spike_mon.i[:])
    assert_equal(spike_times, spike_mon.t_[:])

    # Go back to middle
    net.restore('second')
    assert defaultclock.t == 10*ms
    assert net.t == 10*ms
    net.run(10*ms)
    assert_equal(v_values, state_mon.v[:, :])
    assert_equal(spike_indices, spike_mon.i[:])
    assert_equal(spike_times, spike_mon.t_[:])

    # Go back again (see github issue #681)
    net.restore('second')
    assert defaultclock.t == 10 * ms
    assert net.t == 10 * ms


@attr('codegen-independent')
def test_store_restore_to_file():
    filename = tempfile.mktemp(suffix='state', prefix='brian_test')
    source = NeuronGroup(10, '''dv/dt = rates : 1
                                rates : Hz''', threshold='v>1', reset='v=0')
    source.rates = 'i*100*Hz'
    target = NeuronGroup(10, 'v:1')
    synapses = Synapses(source, target, model='w:1', on_pre='v+=w')
    synapses.connect(j='i')
    synapses.w = 'i*1.0'
    synapses.delay = 'i*ms'
    state_mon = StateMonitor(target, 'v', record=True)
    spike_mon = SpikeMonitor(source)
    net = Network(source, target, synapses, state_mon, spike_mon)
    net.store(filename=filename)  # default time slot
    net.run(10*ms)
    net.store('second', filename=filename)
    net.run(10*ms)
    v_values = state_mon.v[:, :]
    spike_indices, spike_times = spike_mon.it_

    net.restore(filename=filename) # Go back to beginning
    assert defaultclock.t == 0*ms
    assert net.t == 0*ms
    net.run(20*ms)
    assert_equal(v_values, state_mon.v[:, :])
    assert_equal(spike_indices, spike_mon.i[:])
    assert_equal(spike_times, spike_mon.t_[:])

    # Go back to middle
    net.restore('second', filename=filename)
    assert defaultclock.t == 10*ms
    assert net.t == 10*ms
    net.run(10*ms)
    assert_equal(v_values, state_mon.v[:, :])
    assert_equal(spike_indices, spike_mon.i[:])
    assert_equal(spike_times, spike_mon.t_[:])
    try:
        os.remove(filename)
    except OSError:
        pass

@attr('codegen-independent')
def test_store_restore_to_file_new_objects():
    # A more realistic test where the objects are completely re-created
    filename = tempfile.mktemp(suffix='state', prefix='brian_test')
    def create_net():
        # Use a bit of a complicated spike and connection pattern with
        # heterogeneous delays

        # Note: it is important that all objects have the same name, this would
        # be the case if we were running this in a new process but to not rely
        # on garbage collection we will assign explicit names here
        source = SpikeGeneratorGroup(5, np.arange(5).repeat(3),
                                     [3, 4, 1, 2, 3, 7, 5, 4, 1, 0, 5, 9, 7, 8, 9]*ms,
                                     name='source')
        target = NeuronGroup(10, 'v:1', name='target')
        synapses = Synapses(source, target, model='w:1', on_pre='v+=w',
                            name='synapses')
        synapses.connect('j>=i')
        synapses.w = 'i*1.0 + j*2.0'
        synapses.delay = '(5-i)*ms'
        state_mon = StateMonitor(target, 'v', record=True, name='statemonitor')
        input_spikes = SpikeMonitor(source, name='input_spikes')
        net = Network(source, target, synapses, state_mon, input_spikes)
        return net

    net = create_net()
    net.store(filename=filename)  # default time slot
    net.run(5*ms)
    net.store('second', filename=filename)
    net.run(5*ms)
    input_spike_indices = np.array(net['input_spikes'].i)
    input_spike_times = Quantity(net['input_spikes'].t, copy=True)
    v_values_full_sim = Quantity(net['statemonitor'].v[:, :], copy=True)

    net = create_net()
    net.restore(filename=filename)  # Go back to beginning
    net.run(10*ms)
    assert_equal(input_spike_indices, net['input_spikes'].i)
    assert_equal(input_spike_times, net['input_spikes'].t)
    assert_equal(v_values_full_sim, net['statemonitor'].v[:, :])

    net = create_net()
    net.restore('second', filename=filename)  # Go back to middle
    net.run(5*ms)
    assert_equal(input_spike_indices, net['input_spikes'].i)
    assert_equal(input_spike_times, net['input_spikes'].t)
    assert_equal(v_values_full_sim, net['statemonitor'].v[:, :])

    try:
        os.remove(filename)
    except OSError:
        pass


@attr('codegen-independent')
def test_store_restore_to_file_differing_nets():
    # Check that the store/restore mechanism is not used with differing
    # networks
    filename = tempfile.mktemp(suffix='state', prefix='brian_test')

    source = SpikeGeneratorGroup(5, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]*ms,
                                 name='source_1')
    mon = SpikeMonitor(source, name='monitor')
    net = Network(source, mon)
    net.store(filename=filename)

    source_2 = SpikeGeneratorGroup(5, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]*ms,
                                   name='source_2')
    mon = SpikeMonitor(source_2, name='monitor')
    net = Network(source_2, mon)
    assert_raises(KeyError, lambda: net.restore(filename=filename))

    net = Network(source)  # Without the monitor
    assert_raises(KeyError, lambda: net.restore(filename=filename))

@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_store_restore_magic():
    source = NeuronGroup(10, '''dv/dt = rates : 1
                                rates : Hz''', threshold='v>1', reset='v=0')
    source.rates = 'i*100*Hz'
    target = NeuronGroup(10, 'v:1')
    synapses = Synapses(source, target, model='w:1', on_pre='v+=w')
    synapses.connect(j='i')
    synapses.w = 'i*1.0'
    synapses.delay = 'i*ms'
    state_mon = StateMonitor(target, 'v', record=True)
    spike_mon = SpikeMonitor(source)
    store()  # default time slot
    run(10*ms)
    store('second')
    run(10*ms)
    v_values = state_mon.v[:, :]
    spike_indices, spike_times = spike_mon.it_

    restore() # Go back to beginning
    assert magic_network.t == 0*ms
    run(20*ms)
    assert defaultclock.t == 20*ms
    assert_equal(v_values, state_mon.v[:, :])
    assert_equal(spike_indices, spike_mon.i[:])
    assert_equal(spike_times, spike_mon.t_[:])

    # Go back to middle
    restore('second')
    assert magic_network.t == 10*ms
    run(10*ms)
    assert defaultclock.t == 20*ms
    assert_equal(v_values, state_mon.v[:, :])
    assert_equal(spike_indices, spike_mon.i[:])
    assert_equal(spike_times, spike_mon.t_[:])


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_store_restore_magic_to_file():
    filename = tempfile.mktemp(suffix='state', prefix='brian_test')
    source = NeuronGroup(10, '''dv/dt = rates : 1
                                rates : Hz''', threshold='v>1', reset='v=0')
    source.rates = 'i*100*Hz'
    target = NeuronGroup(10, 'v:1')
    synapses = Synapses(source, target, model='w:1', on_pre='v+=w')
    synapses.connect(j='i')
    synapses.w = 'i*1.0'
    synapses.delay = 'i*ms'
    state_mon = StateMonitor(target, 'v', record=True)
    spike_mon = SpikeMonitor(source)
    store(filename=filename)  # default time slot
    run(10*ms)
    store('second', filename=filename)
    run(10*ms)
    v_values = state_mon.v[:, :]
    spike_indices, spike_times = spike_mon.it_

    restore(filename=filename) # Go back to beginning
    assert magic_network.t == 0*ms
    run(20*ms)
    assert defaultclock.t == 20*ms
    assert_equal(v_values, state_mon.v[:, :])
    assert_equal(spike_indices, spike_mon.i[:])
    assert_equal(spike_times, spike_mon.t_[:])

    # Go back to middle
    restore('second', filename=filename)
    assert magic_network.t == 10*ms
    run(10*ms)
    assert defaultclock.t == 20*ms
    assert_equal(v_values, state_mon.v[:, :])
    assert_equal(spike_indices, spike_mon.i[:])
    assert_equal(spike_times, spike_mon.t_[:])
    try:
        os.remove(filename)
    except OSError:
        pass


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_store_restore_spikequeue():
    # See github issue #938
    source = SpikeGeneratorGroup(1, [0], [0] * ms)
    target = NeuronGroup(1, 'v : 1')
    conn = Synapses(source, target, on_pre='v += 1', delay=2 * defaultclock.dt)
    conn.connect()
    run(defaultclock.dt)  # Spike is not yet delivered
    store()
    run(2 * defaultclock.dt)
    assert target.v[0] == 1
    restore()
    run(2 * defaultclock.dt)
    assert target.v[0] == 1
    restore()
    run(2 * defaultclock.dt)
    assert target.v[0] == 1


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_defaultclock_dt_changes():
    BrianLogger.suppress_name('resolution_conflict')
    for dt in [0.1*ms, 0.01*ms, 0.5*ms, 1*ms, 3.3*ms]:
        defaultclock.dt = dt
        G = NeuronGroup(1, 'v:1')
        mon = StateMonitor(G, 'v', record=True)
        net = Network(G, mon)
        net.run(2*dt)
        assert_equal(mon.t[:], [0, dt/ms]*ms)

@attr('standalone-compatible', 'multiple-runs')
@with_setup(teardown=reinit_devices)
def test_dt_changes_between_runs():
    defaultclock.dt = 0.1*ms
    G = NeuronGroup(1, 'v:1')
    mon = StateMonitor(G, 'v', record=True)
    run(.5*ms)
    defaultclock.dt = .5*ms
    run(.5*ms)
    defaultclock.dt = 0.1*ms
    run(.5*ms)
    device.build(direct_call=False, **device.build_options)
    assert len(mon.t[:]) == 5 + 1 + 5
    assert_allclose(mon.t[:],
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1., 1.1, 1.2, 1.3, 1.4]*ms)

@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_dt_restore():
    defaultclock.dt = 0.5*ms
    G = NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1')
    mon = StateMonitor(G, 'v', record=True)
    net = Network(G, mon)
    net.store()

    net.run(1*ms)
    assert_equal(mon.t[:], [0, 0.5]*ms)
    defaultclock.dt = 1*ms
    net.run(2*ms)
    assert_equal(mon.t[:], [0, 0.5, 1, 2]*ms)
    net.restore()
    assert_equal(mon.t[:], [])
    net.run(1*ms)
    assert defaultclock.dt == 0.5*ms
    assert_equal(mon.t[:], [0, 0.5]*ms)


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_continuation():
    defaultclock.dt = 1*ms
    G = NeuronGroup(1, 'dv/dt = -v / (10*ms) : 1')
    G.v = 1
    mon = StateMonitor(G, 'v', record=True)
    net = Network(G, mon)
    net.run(2*ms)

    # Run the same simulation but with two runs that use sub-dt run times
    G2 = NeuronGroup(1, 'dv/dt = -v / (10*ms) : 1')
    G2.v = 1
    mon2 = StateMonitor(G2, 'v', record=True)
    net2 = Network(G2, mon2)
    net2.run(0.5*ms)
    net2.run(1.5*ms)

    assert_equal(mon.t[:], mon2.t[:])
    assert_equal(mon.v[:], mon2.v[:])


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_get_set_states():
    G = NeuronGroup(10, 'v:1', name='a_neurongroup')
    G.v = 'i'
    net = Network(G)
    states1 = net.get_states()
    states2 = magic_network.get_states()
    states3 = net.get_states(read_only_variables=False)
    assert set(states1.keys()) == set(states2.keys()) == set(states3.keys()) == {'a_neurongroup'}
    assert set(states1['a_neurongroup'].keys()) == set(states2['a_neurongroup'].keys()) == {'i', 'dt', 'N', 't', 'v'}
    assert set(states3['a_neurongroup']) == {'v'}

    # Try re-setting the state
    G.v = 0
    net.set_states(states3)
    assert_equal(G.v, np.arange(10))


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_multiple_runs_defaultclock():
    defaultclock.dt = 0.1*ms
    G = NeuronGroup(1, 'dv/dt = -v / (10*ms) : 1')
    net = Network(G)
    net.run(0.5*ms)

    # The new dt is not compatible with the previous time but it should not
    # raise an error because we start a new simulation at time 0
    defaultclock.dt = 1*ms
    G = NeuronGroup(1, 'dv/dt = -v / (10*ms) : 1')
    net = Network(G)
    net.run(1*ms)


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_multiple_runs_defaultclock_incorrect():
    defaultclock.dt = 0.1*ms
    G = NeuronGroup(1, 'dv/dt = -v / (10*ms) : 1')
    net = Network(G)
    net.run(0.5*ms)

    # The new dt is not compatible with the previous time since we cannot
    # continue at 0.5ms with a dt of 1ms
    defaultclock.dt = 1*ms
    assert_raises(ValueError, lambda: net.run(1*ms))


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_profile():
    G = NeuronGroup(10, 'dv/dt = -v / (10*ms) : 1', threshold='v>1',
                    reset='v=0', name='profile_test')
    G.v = 1.1
    net = Network(G)
    net.run(1*ms, profile=True)
    # The should be four simulated CodeObjects, one for the group and one each
    # for state update, threshold and reset + 1 for the clock
    info = net.profiling_info
    info_dict = dict(info)
    # Standalone does not include the NeuronGroup object (which is not doing
    # anything during the run) in the profiling information, while runtime
    # does
    assert 3 <= len(info) <= 4
    assert len(info) == 3 or 'profile_test' in info_dict
    for obj in ['stateupdater', 'thresholder', 'resetter']:
        name = 'profile_test_' + obj
        assert name in info_dict or name + '_codeobject' in info_dict
    assert all([t>=0*second for _, t in info])


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_profile_off():
    G = NeuronGroup(10, 'dv/dt = -v / (10*ms) : 1', threshold='v>1',
                    reset='v=0', name='profile_test')
    net = Network(G)
    net.run(1*ms, profile=False)
    assert_raises(ValueError, lambda: profiling_summary(net))


@attr('codegen-independent')
def test_profile_ipython_html():
    G = NeuronGroup(10, 'dv/dt = -v / (10*ms) : 1', threshold='v>1',
                    reset='v=0', name='profile_test')
    G.v = 1.1
    net = Network(G)
    net.run(1*ms, profile=True)
    summary = profiling_summary(net)
    assert len(summary._repr_html_())


@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_magic_scope():
    '''
    Check that `start_scope` works as expected.
    '''
    G1 = NeuronGroup(1, 'v:1', name='G1')
    G2 = NeuronGroup(1, 'v:1', name='G2')
    objs1 = {obj.name for obj in collect()}
    start_scope()
    G3 = NeuronGroup(1, 'v:1', name='G3')
    G4 = NeuronGroup(1, 'v:1', name='G4')
    objs2 = {obj.name for obj in collect()}
    assert objs1=={'G1', 'G2'}
    assert objs2=={'G3', 'G4'}


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_runtime_rounding():
    # Test that runtime and standalone round in the same way, see github issue
    # #695 for details
    defaultclock.dt = 20.000000000020002 * us
    G = NeuronGroup(1, 'v:1')
    mon = StateMonitor(G, 'v', record=True)
    run(defaultclock.dt * 250)
    assert len(mon.t) == 250

@attr('codegen-independent')
def test_small_runs():
    # One long run and multiple small runs should give the same results
    group_1 = NeuronGroup(10, 'dv/dt = -v / (10*ms) : 1')
    group_1.v = '(i + 1) / N'
    mon_1 = StateMonitor(group_1, 'v', record=True)
    net_1 = Network(group_1, mon_1)
    net_1.run(1*second)

    group_2 = NeuronGroup(10, 'dv/dt = -v / (10*ms) : 1')
    group_2.v = '(i + 1) / N'
    mon_2 = StateMonitor(group_2, 'v', record=True)
    net_2 = Network(group_2, mon_2)
    runtime = 1*ms
    while True:
        runtime *= 3
        runtime = min([runtime, 1*second - net_2.t])
        net_2.run(runtime)
        if net_2.t >= 1*second:
            break

    assert_allclose(mon_1.t_[:], mon_2.t_[:])
    assert_allclose(mon_1.v_[:], mon_2.v_[:])


@attr('codegen-independent')
@with_setup(teardown=reinit_devices)
def test_long_run_dt_change():
    # Check that the dt check is not too restrictive, see issue #730 for details
    group = NeuronGroup(1, '')  # does nothing...
    defaultclock.dt = 0.1*ms
    run(100*second)
    # print profiling_summary()
    defaultclock.dt = 0.01*ms
    run(1*second)

@attr('standalone-compatible', 'multiple-runs')
@with_setup(teardown=reinit_devices)
def test_multiple_runs_constant_change():
    const_v = 1
    group = NeuronGroup(1, 'v = const_v : 1')
    mon = StateMonitor(group, 'v', record=0)
    run(defaultclock.dt)
    const_v = 2
    run(defaultclock.dt)
    device.build(direct_call=False, **device.build_options)
    assert_equal(mon.v[0], [1, 2])


@attr('standalone-compatible', 'multiple-runs')
@with_setup(teardown=reinit_devices)
def test_multiple_runs_function_change():
    inp = TimedArray([1, 2], dt=defaultclock.dt)
    group = NeuronGroup(1, 'v = inp(t) : 1')
    mon = StateMonitor(group, 'v', record=0)
    run(2*defaultclock.dt)
    inp = TimedArray([0, 0, 3, 4], dt=defaultclock.dt)
    run(2*defaultclock.dt)
    device.build(direct_call=False, **device.build_options)
    assert_equal(mon.v[0], [1, 2, 3, 4])

if __name__ == '__main__':
    BrianLogger.log_level_warn()
    for t in [
            test_incorrect_network_use,
            test_network_contains,
            test_empty_network,
            test_network_single_object,
            test_network_two_objects,
            test_network_from_dict,
            test_network_different_clocks,
            test_network_different_when,
            test_network_default_schedule,
            test_network_schedule_change,
            test_network_before_after_schedule,
            test_network_custom_slots,
            test_network_incorrect_schedule,
            test_schedule_warning,
            test_scheduling_summary_magic,
            test_scheduling_summary,
            test_magic_network,
            test_network_stop,
            test_network_operations,
            test_incorrect_network_operations,
            test_network_active_flag,
            test_network_t,
            test_incorrect_dt_defaultclock,
            test_incorrect_dt_custom_clock,
            test_network_remove,
            test_magic_weak_reference,
            test_magic_unused_object,
            test_invalid_magic_network,
            test_multiple_networks_invalid,
            test_network_access,
            test_loop,
            test_magic_collect,
            test_progress_report,
            test_progress_report_incorrect,
            test_multiple_runs_report_standalone,
            test_multiple_runs_report_standalone_2,
            test_multiple_runs_report_standalone_3,
            test_multiple_runs_report_standalone_incorrect,
            test_store_restore,
            test_store_restore_to_file,
            test_store_restore_to_file_new_objects,
            test_store_restore_to_file_differing_nets,
            test_store_restore_magic,
            test_store_restore_magic_to_file,
            test_store_restore_spikequeue,
            test_defaultclock_dt_changes,
            test_dt_changes_between_runs,
            test_dt_restore,
            test_continuation,
            test_get_set_states,
            test_multiple_runs_defaultclock,
            test_multiple_runs_defaultclock_incorrect,
            test_profile,
            test_profile_off,
            test_profile_ipython_html,
            test_magic_scope,
            test_runtime_rounding,
            test_small_runs,
            test_long_run_dt_change,
            test_multiple_runs_constant_change,
            test_multiple_runs_function_change
            ]:
        set_device(all_devices['runtime'])
        t()
        reinit_devices()
