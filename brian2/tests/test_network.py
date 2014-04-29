import copy
import logging
import weakref

import numpy as np
from numpy.testing import assert_equal, assert_raises
from nose import with_setup

from brian2 import (Clock, Network, ms, second, BrianObject, defaultclock,
                    run, stop, NetworkOperation, network_operation,
                    restore_initial_state, MagicError, clear, Synapses,
                    NeuronGroup, StateMonitor, SpikeMonitor,
                    PopulationRateMonitor, MagicNetwork, magic_network)
from brian2.utils.proxy import Proxy
from brian2.utils.logger import catch_logs, BrianLogger
from brian2.tests import expected_python3_failure

def test_incorrect_network_use():
    '''Test some wrong uses of `Network` and `MagicNetwork`'''
    assert_raises(TypeError, lambda: Network('not a BrianObject'))
    net = Network()
    assert_raises(TypeError, lambda: net.add('not a BrianObject'))
    assert_raises(ValueError, lambda: MagicNetwork())
    G = NeuronGroup(10, 'v:1')
    assert_raises(MagicError, lambda: magic_network.add(G))
    assert_raises(MagicError, lambda: magic_network.remove(G))


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


@with_setup(teardown=restore_initial_state)
def test_network_single_object():
    # Check that a network with a single object functions correctly
    x = Counter()
    net = Network(x)
    net.run(1*ms)
    assert_equal(x.count, 10)

@with_setup(teardown=restore_initial_state)
def test_network_two_objects():
    # Check that a network with two objects and the same clock function correctly
    x = Counter(when=5)
    y = Counter(when=6)
    net = Network()
    net.add([x, [y]]) # check that a funky way of adding objects work correctly
    assert_equal(net.objects[0].order, 5)
    assert_equal(net.objects[1].order, 6)
    assert_equal(len(net.objects), 2)
    net.run(1*ms)
    assert_equal(x.count, 10)
    assert_equal(y.count, 10)

updates = []
class NameLister(BrianObject):
    add_to_magic_network = True
    def __init__(self, **kwds):
        super(NameLister, self).__init__(**kwds)

    def run(self):
        updates.append(self.name)

@with_setup(teardown=restore_initial_state)
def test_network_different_clocks():
    # Check that a network with two different clocks functions correctly
    clock1 = Clock(dt=1*ms)
    clock3 = Clock(dt=3*ms)
    x = NameLister(name='x', when=(clock1, 0))
    y = NameLister(name='y', when=(clock3, 1))
    net = Network(x, y)
    net.run(10*ms)
    assert_equal(''.join(updates), 'xyxxxyxxxyxxxy')

@with_setup(teardown=restore_initial_state)
def test_network_different_when():
    # Check that a network with different when attributes functions correctly
    updates[:] = []
    x = NameLister(name='x', when='start')
    y = NameLister(name='y', when='end')
    net = Network(x, y)
    net.run(0.3*ms)
    assert_equal(''.join(updates), 'xyxyxy')

class Preparer(BrianObject):
    add_to_magic_network = True
    def __init__(self, **kwds):
        super(Preparer, self).__init__(**kwds)
        self.did_reinit = False
        self.did_pre_run = False
        self.did_post_run = False
    def reinit(self):
        self.did_reinit = True
    def before_run(self, namespace=None, level=0):
        self.did_pre_run = True
    def after_run(self):
        self.did_post_run = True        

@with_setup(teardown=restore_initial_state)
def test_network_reinit_pre_post_run():
    # Check that reinit and before_run and after_run work
    x = Preparer()
    net = Network(x)
    assert_equal(x.did_reinit, False)
    assert_equal(x.did_pre_run, False)
    assert_equal(x.did_post_run, False)
    net.run(1*ms)
    assert_equal(x.did_reinit, False)
    assert_equal(x.did_pre_run, True)
    assert_equal(x.did_post_run, True)
    net.reinit()
    assert_equal(x.did_reinit, True)

    # Make sure that running with "report" works
    net.run(1*ms, report='stdout')

@with_setup(teardown=restore_initial_state)
def test_magic_network():
    # test that magic network functions correctly
    x = Counter()
    y = Counter()
    run(10*ms)
    assert_equal(x.count, 100)
    assert_equal(y.count, 100)

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

@with_setup(teardown=restore_initial_state)
def test_network_stop():
    # test that Network.stop and global stop() work correctly
    net = Network()
    x = Stopper(10, net.stop)
    net.add(x)
    net.run(10*ms)
    assert_equal(defaultclock.t, 1*ms)
    
    del net
    defaultclock.t = 0*second
    
    x = Stopper(10, stop)
    net = Network(x)
    net.run(10*ms)
    assert_equal(defaultclock.t, 1*ms)

@with_setup(teardown=restore_initial_state)
def test_network_operations():
    # test NetworkOperation and network_operation
    seq = []
    def f1():
        seq.append('a')
    op1 = NetworkOperation(f1, when=('start', 1))
    @network_operation
    def f2():
        seq.append('b')
    @network_operation(when=('end', 1))
    def f3():
        seq.append('c')
    run(1*ms)
    assert_equal(''.join(seq), 'bac'*10)

@with_setup(teardown=restore_initial_state)
def test_network_active_flag():
    # test that the BrianObject.active flag is recognised by Network.run
    x = Counter()
    y = Counter()
    y.active = False
    run(1*ms)
    assert_equal(x.count, 10)
    assert_equal(y.count, 0)

@with_setup(teardown=restore_initial_state)
def test_network_t():
    # test that Network.t works as expected
    c1 = Clock(dt=1*ms)
    c2 = Clock(dt=2*ms)
    x = Counter(when=c1)
    y = Counter(when=c2)
    net = Network(x, y)
    net.run(4*ms)
    assert_equal(c1.t, 4*ms)
    assert_equal(c2.t, 4*ms)
    assert_equal(net.t, 4*ms)
    net.run(1*ms)
    assert_equal(c1.t, 5*ms)
    assert_equal(c2.t, 6*ms)
    assert_equal(net.t, 5*ms)
    assert_equal(x.count, 5)
    assert_equal(y.count, 3)
    net.run(0.5*ms) # should only update x
    assert_equal(c1.t, 6*ms)
    assert_equal(c2.t, 6*ms)
    assert_equal(net.t, 5.5*ms)
    assert_equal(x.count, 6)
    assert_equal(y.count, 3)
    net.run(0.5*ms) # shouldn't do anything
    assert_equal(c1.t, 6*ms)
    assert_equal(c2.t, 6*ms)
    assert_equal(net.t, 6*ms)
    assert_equal(x.count, 6)
    assert_equal(y.count, 3)
    net.run(0.5*ms) # should update x and y
    assert_equal(c1.t, 7*ms)
    assert_equal(c2.t, 8*ms)
    assert_equal(net.t, 6.5*ms)
    assert_equal(x.count, 7)
    assert_equal(y.count, 4)
    
    del c1, c2, x, y, net

    # now test with magic run    
    c1 = Clock(dt=1*ms)
    c2 = Clock(dt=2*ms)
    x = Counter(when=c1)
    y = Counter(when=c2)
    run(4*ms)
    assert_equal(c1.t, 4*ms)
    assert_equal(c2.t, 4*ms)
    assert_equal(x.count, 4)
    assert_equal(y.count, 2)
    run(4*ms)
    assert_equal(c1.t, 8*ms)
    assert_equal(c2.t, 8*ms)
    assert_equal(x.count, 8)
    assert_equal(y.count, 4)
    run(1*ms)
    assert_equal(c1.t, 9*ms)
    assert_equal(c2.t, 10*ms)
    assert_equal(x.count, 9)
    assert_equal(y.count, 5)
    
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

@with_setup(teardown=restore_initial_state)
def test_network_copy():
    x = Counter()
    net = Network(x)
    net2 = Network()
    for obj in net.objects:
        net2.add(obj)
    net2.run(1*ms)
    assert_equal(x.count, 10)
    net.run(1*ms)
    assert_equal(x.count, 20)

class NoninvalidatingCounter(Counter):
    add_to_magic_network = True
    invalidates_magic_network = False

@with_setup(teardown=restore_initial_state)
def test_invalid_magic_network():
    x = Counter()
    run(1*ms)
    assert_equal(x.count, 10)
    y = Counter()
    assert_raises(MagicError, lambda: run(1*ms, level=3))
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
    del x
    assert_raises(MagicError, lambda: run(1*ms, level=3))
    clear()
    z = Counter()
    run(1*ms)
    assert_equal(z.count, 10)
    assert_equal(magic_network.t, 1*ms)


@with_setup(teardown=restore_initial_state)
def test_magic_weak_reference():
    '''Test that holding a weak reference to an object does not make it get '''

    G1 = NeuronGroup(1, 'v:1')

    # this object should not be included
    G2 = weakref.ref(NeuronGroup(1, 'v:1'))

    with catch_logs(log_level=logging.DEBUG) as l:
        run(1*ms)

        # Check the debug messages for the number of included objects
        magic_objects = [msg[2] for msg in l
                         if msg[1] == 'brian2.core.magic.magic_objects'][0]
        assert '2 objects' in magic_objects, 'Unexpected log message: %s' % magic_objects


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


@with_setup(teardown=restore_initial_state)
def test_network_access():
    x = Counter(name='counter')
    net = Network(x)
    assert len(net) == 1

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


@with_setup(teardown=restore_initial_state)
@expected_python3_failure
def test_dependency_check():
    def create_net():
        G = NeuronGroup(10, 'v: 1')
        dependent_objects = [
                             StateMonitor(G, 'v', record=True),
                             SpikeMonitor(G),
                             PopulationRateMonitor(G),
                             Synapses(G, G, pre='v+=1', connect=True)
                             ]
        return dependent_objects

    dependent_objects = create_net()
    # Trying to simulate the monitors/synapses without the group should fail
    for obj in dependent_objects:
        assert_raises(ValueError, lambda: Network(obj).run(0*ms))

    # simulation with a magic network should work, but all objects should be
    # inactive
    assert all(obj.active for obj in dependent_objects)
    run(0*ms)
    assert all(not obj.active for obj in dependent_objects)


@with_setup(teardown=restore_initial_state)
@expected_python3_failure
def test_proxy():
    '''
    Make sure that `Proxy` objects do not make objects get included in `MagicNetwork`.
    '''
    updates[:] = []
    x = NameLister(name='x')
    y = NameLister(name='y')
    p = Proxy(y)
    del y
    run(defaultclock.dt)
    # Object y should not have been run
    assert updates == ['x']
    # Proxy should still work
    assert p.name == 'y'


@with_setup(teardown=restore_initial_state)
@expected_python3_failure
def test_loop_with_proxies():
    '''
    Somewhat realistic test with a loop of magic networks and proxy objects
    '''
    updates[:] = []
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



@with_setup(teardown=restore_initial_state)
@expected_python3_failure
def test_loop_with_proxies_2():
    '''
    Somewhat realistic test with a loop of magic networks and proxy objects
    '''
    updates[:] = []
    def run_simulation():
        G = NeuronGroup(10, 'dv/dt = -v / (10*ms) : 1',
                        reset='v=0', threshold='v>1')
        G.v = np.linspace(0, 1, 10)
        v_mon = StateMonitor(G, 'v', record=True)
        spike_mon = SpikeMonitor(G)
        r_mon = PopulationRateMonitor(G)
        run(1*ms)
        # We return potentially problematic references to monitors
        return v_mon, spike_mon, r_mon

    # First run
    with catch_logs(log_level=logging.DEBUG) as l:
        v_mon, spike_mon, r_mon = run_simulation()
        # Check the debug messages for the number of included objects
        magic_objects = [msg[2] for msg in l
                         if msg[1] == 'brian2.core.magic.magic_objects'][0]
        assert '7 objects' in magic_objects
    # all monitors should be active
    assert v_mon.active and spike_mon.active and r_mon.active

    # Second run (we should get an information for each monitor)
    with catch_logs(log_level=logging.DEBUG) as l:
        run_simulation()
        missing_dependencies = len([msg for msg in l
                                    if msg[1] == 'brian2.core.magic.missing_dependency'])
        assert missing_dependencies == 3
        # Check the debug messages for the number of included objects -- note
        # that this still includes the monitors from the previous run. These
        # will be set to inactive and not simulated, though.
        magic_objects = [msg[2] for msg in l
                         if msg[1] == 'brian2.core.magic.magic_objects'][0]
        assert '10 objects' in magic_objects

        # The monitors from the previous run should be all inactive now
        assert not (v_mon.active or spike_mon.active or r_mon.active)


if __name__=='__main__':
    for t in [
              test_incorrect_network_use,
              test_empty_network,
              test_network_single_object,
              test_network_two_objects,
              test_network_different_clocks,
              test_network_different_when,
              test_network_reinit_pre_post_run,
              test_magic_network,
              test_network_stop,
              test_network_operations,
              test_network_active_flag,
              test_network_t,
              test_network_remove,
              test_network_copy,
              test_magic_weak_reference,
              test_magic_unused_object,
              test_invalid_magic_network,
              test_network_access,
              test_dependency_check,
              test_proxy,
              test_loop_with_proxies,
              test_loop_with_proxies_2
              ]:
        t()
        restore_initial_state()
