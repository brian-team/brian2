from brian2 import *
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
    assert_raises(ValueError, lambda: BrianObject(name='brianobject'))

if __name__=='__main__':
    test_base()
    test_names()
