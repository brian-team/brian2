from brian2 import *
from numpy.testing import assert_raises, assert_equal
from nose import with_setup

class DerivedBrianObject(BrianObject):
    def __init__(self, name):
        super(DerivedBrianObject, self).__init__()
        self.name = name
    def __str__(self):
        return self.name
    __repr__ = __str__

def namesof(cls):
    return ''.join(sorted(map(str, get_instances(cls))))

def makesomething():
    return DerivedBrianObject('z')

class MoreDerivedBrianObject1(DerivedBrianObject):
    pass

class MoreDerivedBrianObject2(DerivedBrianObject):
    pass

@with_setup(teardown=restore_initial_state)
def test_base():    
    x = DerivedBrianObject('x')
    assert_equal(namesof(DerivedBrianObject), 'x')
    y = DerivedBrianObject('y')
    assert_equal(namesof(DerivedBrianObject), 'xy')
    del y
    assert_equal(namesof(DerivedBrianObject), 'x')
    
    
    makesomething()
    assert_equal(namesof(DerivedBrianObject), 'x')
    z = makesomething()
    assert_equal(namesof(DerivedBrianObject), 'xz')
    
    a = MoreDerivedBrianObject1('a')
    b = MoreDerivedBrianObject2('b')
    
    assert_equal(namesof(DerivedBrianObject), 'abxz')
    assert_equal(namesof(MoreDerivedBrianObject1), 'a')
    assert_equal(namesof(MoreDerivedBrianObject2), 'b')
    
    x = DerivedBrianObject('x')
    y = DerivedBrianObject('y')
    assert_equal(x.when, 'start')
    assert_equal(x.order, 0)
    assert_equal(len(x.contained_objects), 0)
    assert_raises(AttributeError, lambda: setattr(x, 'contained_objects', []))
    x.contained_objects.append(y)
    assert_equal(len(x.contained_objects), 1)
    assert x.contained_objects[0] is y
    
    assert x.clock is defaultclock
    
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

if __name__=='__main__':
    test_base()
