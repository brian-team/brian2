from collections import defaultdict
from weakref import ref

__all__ = ["Trackable"]


class InstanceTrackerSet(set):
    """
    A `set` of `weakref.ref` to all existing objects of a certain class.

    Should not normally be directly used.
    """

    def add(self, value):
        """
        Adds a `weakref.ref` to the ``value``
        """
        # The second argument to ref is a callback that is called with the
        # ref as argument when the object has been deleted, here we just
        # remove it from the set in that case
        wr = ref(value, self.remove)
        set.add(self, wr)

    def remove(self, value):
        """
        Removes the ``value`` (which should be a weakref) if it is in the set

        Sometimes the value will have been removed from the set by `clear`,
        so we ignore `KeyError` in this case.
        """
        try:
            set.remove(self, value)
        except KeyError:
            pass


class InstanceFollower(object):
    """
    Keep track of all instances of classes derived from `Trackable`

    The variable ``__instancesets__`` is a dictionary with keys which are class
    objects, and values which are `InstanceTrackerSet`, so
    ``__instanceset__[cls]`` is a set tracking all of the instances of class
    ``cls`` (or a subclass).
    """

    instance_sets = defaultdict(InstanceTrackerSet)

    def add(self, value):
        for (
            cls
        ) in (
            value.__class__.__mro__
        ):  # MRO is the Method Resolution Order which contains all the superclasses of a class
            self.instance_sets[cls].add(value)

    def get(self, cls):
        return self.instance_sets[cls]


class Trackable(object):
    """
    Classes derived from this will have their instances tracked.

    The `classmethod` `__instances__()` will return an `InstanceTrackerSet`
    of the instances of that class, and its subclasses.
    """

    __instancefollower__ = (
        InstanceFollower()
    )  # static property of all objects of class derived from Trackable

    def __new__(typ, *args, **kw):
        obj = object.__new__(typ)
        obj.__instancefollower__.add(obj)
        return obj

    @classmethod
    def __instances__(cls):
        return cls.__instancefollower__.get(cls)
