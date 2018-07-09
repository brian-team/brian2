from brian2 import *
from brian2.devices.device import reinit_devices
from brian2.tests.utils import assert_allclose

from numpy.testing import assert_raises, assert_equal
from nose import with_setup
from nose.plugins.attrib import attr


class DerivedBrianObject(BrianObject):
    def __init__(self, name='derivedbrianobject*'):
        super(DerivedBrianObject, self).__init__(name=name)
    def __str__(self):
        return self.name
    __repr__ = __str__

@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_base():
    x = DerivedBrianObject('x')
    y = DerivedBrianObject('y')
    assert_equal(x.when, 'start')
    assert_equal(x.order, 0)
    assert_equal(len(x.contained_objects), 0)
    assert_raises(AttributeError, lambda: setattr(x, 'contained_objects', []))
    x.contained_objects.append(y)
    assert_equal(len(x.contained_objects), 1)
    assert x.contained_objects[0] is y

    assert_equal(x.active, True)
    assert_equal(y.active, True)
    y.active = False
    assert_equal(x.active, True)
    assert_equal(y.active, False)
    y.active = True
    assert_equal(x.active, True)
    assert_equal(y.active, True)
    x.active = False
    assert_equal(x.active, False)
    assert_equal(y.active, False)

@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_names():
    obj = BrianObject()
    obj2 = BrianObject()
    obj3 = DerivedBrianObject()
    assert_equal(obj.name, 'brianobject')
    assert_equal(obj2.name, 'brianobject_1')
    assert_equal(obj3.name, 'derivedbrianobject')

@attr('codegen-independent')
@with_setup(teardown=restore_initial_state)
def test_duplicate_names():
    # duplicate names are allowed, as long as they are not part of the
    # same network
    obj1 = BrianObject(name='name1')
    obj2 = BrianObject(name='name2')
    obj3 = BrianObject(name='name')
    obj4 = BrianObject(name='name')
    net = Network(obj1, obj2)
    # all is good
    net.run(0*ms)
    net = Network(obj3, obj4)
    assert_raises(ValueError, lambda: net.run(0*ms))


@attr('standalone-compatible', 'multiple-runs')
@with_setup(teardown=reinit_devices)
def test_active_flag():
    G = NeuronGroup(1, 'dv/dt = 1/ms : 1')
    mon = StateMonitor(G, 'v', record=0)
    mon.active = False
    run(1*ms)
    mon.active = True
    G.active = False
    run(1*ms)
    device.build(direct_call=False, **device.build_options)
    # Monitor should start recording at 1ms
    # Neurongroup should not integrate after 1ms (but should have integrated before)
    assert_allclose(mon[0].t[0], 1*ms)
    assert_allclose(mon[0].v, 1.0)


if __name__=='__main__':
    test_base()
    test_names()
    test_duplicate_names()
    test_active_flag()
