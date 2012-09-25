from brian2 import *
from numpy.testing import assert_raises, assert_equal

class DerivedBrianObject(BrianObject):
    def __init__(self, name):
        super(DerivedBrianObject, self).__init__()
        self.name = name
    def __str__(self):
        return self.name
    __repr__ = __str__

def namesof(cls):
    return ''.join(sorted(map(str, get_instances(cls))))
    
x = DerivedBrianObject('x')
assert_equal(namesof(DerivedBrianObject), 'x')
y = DerivedBrianObject('y')
assert_equal(namesof(DerivedBrianObject), 'xy')
del y
assert_equal(namesof(DerivedBrianObject), 'x')

def makesomething():
    return DerivedBrianObject('z')

makesomething()
assert_equal(namesof(DerivedBrianObject), 'x')
z = makesomething()
assert_equal(namesof(DerivedBrianObject), 'xz')

class MoreDerivedBrianObject1(DerivedBrianObject):
    pass

class MoreDerivedBrianObject2(DerivedBrianObject):
    pass

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
