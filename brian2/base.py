'''
All Brian objects should derive from :class:`BrianObject`.
'''

from weakref import ref
import clocks

__all__ = ['BrianObject', 'get_instances', 'InstanceTracker']


class WeakSet(set):
    """A set of extended references
    
    Removes references from the set when they are destroyed."""
    def add(self, value):
        wr = ref(value, self.remove)
        set.add(self, wr)

    def get(self):
        return [x() for x in self]


class InstanceFollower(object):
    """Keep track of all instances of classes derived from `InstanceTracker`
    
    The variable __instancesets__ is a dictionary with keys which are class
    objects, and values which are WeakSets, so __instanceset__[cls] is a
    weak set tracking all of the instances of class cls (or a subclass).
    """
    __instancesets__ = {}
    def add(self, value):
        for cls in value.__class__.__mro__: # MRO is the Method Resolution Order which contains all the superclasses of a class
            if cls not in self.__instancesets__:
                self.__instancesets__[cls] = WeakSet()
            self.__instancesets__[cls].add(value)

    def get(self, cls):
        if not cls in self.__instancesets__: return []
        return self.__instancesets__[cls].get()


class InstanceTracker(object):
    """Base class for all classes whose instances are to be tracked
    
    Derive your class from this one to automagically keep track of instances of
    it. If you want a subclass of a tracked class not to be tracked, define the
    attribute ``_track_instances=False``.
    """
    __instancefollower__ = InstanceFollower() # static property of all objects of class derived from InstanceTracker
    _track_instances = True

    def __new__(cls, *args, **kw):
        obj = object.__new__(cls)
        if obj._track_instances:
            obj.__instancefollower__.add(obj)
        return obj


def get_instances(instancetype):
    '''
    Return all instances of a given `InstanceTracker` derived class.
    '''
    try:
        follower = instancetype.__instancefollower__
    except AttributeError:
        raise TypeError('Cannot track instances of class '+str(instancetype.__name__))
    return follower.get(instancetype)

    
class BrianObject(InstanceTracker):
    '''
    All Brian objects derive from this class, defines magic tracking and update.
    
    Parameters
    ----------
    when : str, optional
        Defines when the object is updated in the main :meth:`Network.run`
        loop.
    order : (int, float), optional
        Objects with the same ``when`` value will be updated in order
        of increasing values of ``order``, or if both are equal then the order
        is unspecified (but will always be the same on each iteration).
    clock : `Clock`, optional
        The update clock determining when the object will be updated, or
        use the default clock if unspecified.

    Notes
    -----
    
    See the documentation for :meth:`Network` for an explanation of which
    objects get updated in which order.
    
    The list of all instances of a particular class and its derived classes
    can be returned using the :func:`get_instances` function.
    
    Brian objects deriving from this class should always define an
    ``update()`` method, that gets called by :meth:`Network.run`.
    '''
    
    def __init__(self, when='start', order=0, clock=None):
        if not isinstance(when, str):
            raise TypeError("when attribute should be a string, was "+repr(when))
        if clock is None:
            clock = clocks.defaultclock
        if not isinstance(clock, clocks.Clock):
            raise TypeError("clock should have type Clock, was "+clock.__class__.__name__)
     
        #: The ID string determining when the object should be updated in :meth:`Network.run`.   
        self.when = when
        
        #: The order in which objects with the same clock and ``when`` should be updated
        self.order = order
        
#        #: The `Clock` determining when the object should be updated.
#        self.clock = clock
        self._clock = clock
        
        self._contained_objects = []
        
    def prepare(self):
        '''
        Optional method to prepare data for the first time.
        
        Called by :meth:`Network.prepare`. Note that this method will not be
        called until just before the Network is about to be run, but may be
        called more than once even if the object has already been prepared, so
        the class should keep track of whether it has already been prepared or
        not.
        '''
        pass
        
    def update(self):
        '''
        Every `BrianObject` should define an ``update()`` method which is called every time step.
        '''
        raise NotImplementedError("Classes deriving from BrianObject must "
                                  "define an update() method.")
        
    def reinit(self):
        '''
        Reinitialise the object, called by :meth:`Network.reinit`.
        '''
        pass

    contained_objects = property(fget=lambda self:self._contained_objects,
                                 doc='''
         The list of objects contained within the `BrianObject`.
         
         When a `BrianObject` is added to a `Network`, its contained objects will
         be added as well. This allows for compound objects which contain
         a mini-network structure.
         
         Note that this attribute cannot be set directly, you need to modify
         the underlying list, e.g. ``obj.contained_objects.extend([A, B])``.
         ''')
    
    clock = property(fget=lambda self: self._clock,
                     doc='''
                     The `Clock` determining when the object should be updated.
                     
                     Note that this cannot be changed after the object is
                     created.
                     ''')
