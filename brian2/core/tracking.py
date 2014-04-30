import gc
from collections import defaultdict
from weakref import ref, WeakSet

__all__ = ['Trackable']


class InstanceFollower(object):
    """
    Keep track of all instances of classes derived from `Trackable`
    
    The variable ``__instancesets__`` is a dictionary with keys which are class
    objects, and values which are `WeakSet` objects, so
    ``__instanceset__[cls]`` is a set tracking all of the instances of class
    ``cls`` (or a subclass).
    """
    instance_sets = defaultdict(WeakSet)
    def add(self, value):
        for cls in value.__class__.__mro__: # MRO is the Method Resolution Order which contains all the superclasses of a class
            self.instance_sets[cls].add(value)

    def get(self, cls):
        return self.instance_sets[cls]


class Trackable(object):
    '''
    Classes derived from this will have their instances tracked.
    
    The `classmethod` `__instances__()` will return a `WeakSet`
    of the instances of that class, and its subclasses.
    '''
    __instancefollower__ = InstanceFollower() # static property of all objects of class derived from Trackable
    def __new__(typ, *args, **kw):
        obj = object.__new__(typ)
        obj.__instancefollower__.add(obj)
        return obj

    @classmethod
    def __instances__(cls):
        # Make sure we don't have any objects that are only still present
        # because of cyclic references
        gc.collect()
        return cls.__instancefollower__.get(cls)
