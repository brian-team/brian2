'''
All Brian objects should derive from :class:`BrianObject`.
'''

from weakref import ref, proxy
import gc
import copy

import brian2.core.clocks as clocks

__all__ = ['BrianObject',
           'MagicError',
           'BrianObjectSet',
           'brian_objects',
           'clear',
           ]


class MagicError(Exception):
    '''
    Error that is raised when something goes wrong in `MagicNetwork`
    
    See notes to `MagicNetwork` for more details.
    '''
    pass


class BrianObjectSet(set):
    '''
    A `set` of weak proxies to all existing `BrianObject` objects
    
    Notes
    -----
    
    Has three possible `state` values, `STATE_NEW` (new), `STATE_VALID` (valid) and
    `STATE_INVALID` (invalid). When the set is empty, the state is set to new.
    When a `BrianObject` with `BrianObject.invalidates_magic_network` set to
    ``True`` is added or removed, if the current state is valid it will be
    set to invalid (nothing happens if it is new). When iterating over the set
    if the state is new it is set to valid, and if it is invalid a
    `MagicError` will be raised. You can explicitly set the `state` to
    `STATE_VALID` however.
    
    Iterating through this returns `weakref.proxy` objects of the items in the
    set.
    
    See `run` and `MagicNetwork` for the rationale behind these state
    transitions.
    
    Raises
    ------
    
    MagicError
        If you attempt to iterate through the set when it is in the invalid
        state. See `run` and `MagicNetwork` for more details.
    '''
    
    #: State entered when set is empty
    STATE_NEW = 0
    #: State entered when set is used validly
    STATE_VALID = 1
    #: State entered when an invalidating `BrianObject` is added or removed
    STATE_INVALID = 2
    
    def __init__(self):
        super(BrianObjectSet, self).__init__()
        
        #: Current validity state (one of `STATE_NEW`, `STATE_VALID` or `STATE_INVALID`)
        self.state = self.STATE_NEW
        
    def add(self, value):
        # For the callback to remove the object from the set,
        # we have to remove either with the invalidating or non-invalidating
        # form of remove, dependent on value.invalidates_magic_network. We
        # can't have a simple function remove because by the time the
        # weakref callback is called the object has already been deleted so
        # we can't access its 
        if value.invalidates_magic_network:
            wr = ref(value, self.remove_invalidates)
        else:
            wr = ref(value, self.remove_valid)
        set.add(self, wr)
        if self.state==self.STATE_VALID:
            if value.invalidates_magic_network:
                self.state = self.STATE_INVALID

    def remove(self, value, invalidates_magic_network):
        set.remove(self, value)
        if len(self)==0:
            self.state = self.STATE_NEW
        elif self.state==self.STATE_VALID:
            if invalidates_magic_network:
                self.state = self.STATE_INVALID
                
    def remove_invalidates(self, value):
        self.remove(value, True)

    def remove_valid(self, value):
        self.remove(value, False)
    
    def __iter__(self):
        if self.state==self.STATE_NEW:
            self.state = self.STATE_VALID
        elif self.state==self.STATE_INVALID:
            raise MagicError("Cannot iterate through invalid BrianObjectSet")
        return [proxy(obj()) for obj in set.__iter__(self)].__iter__()
    
    def clear(self):
        set.clear(self)
        self.state = self.STATE_NEW


#: 'A `BrianObjectSet` containing all instances of `BrianObject`
brian_objects = BrianObjectSet()


class BrianObject(object):
    '''
    All Brian objects derive from this class, defines magic tracking and update.

    See the documentation for `Network` for an explanation of which
    objects get updated in which order.
    
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
        
    The set of all `BrianObject`\ s is stored in `brian_objects`.
    
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
        
        self._active = True

    #: Whether or not `MagicNetwork` is invalidated when a new `BrianObject` of this type is created or removed
    invalidates_magic_network = True

    def __new__(cls, *args, **kw):
        obj = object.__new__(cls)
        brian_objects.add(obj)
        return obj
    
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
    
    def _set_active(self, val):
        val = bool(val)
        self._active = val
        for obj in self.contained_objects:
            obj.active = val

    active = property(fget=lambda self:self._active,
                      fset=_set_active,
                      doc='''
                        Whether or not the object should be run.
                        
                        Inactive objects will not have their `update`
                        method called in `Network.run`. Note that setting or
                        unsetting the `active` attribute will set or unset
                        it for all `contained_objects`. 
                        ''')

    
def clear(erase=False):
    '''
    Stops all Brian objects from being automatically detected

    Stops objects from being tracked by `run` and `reinit`.
    Use this if you are seeing `MagicError` on repeated runs.
    
    Parameters
    ----------
    
    erase : bool, optional
        If set to ``True``, all data attributes of all Brian objects
        will be set to ``None``. This
        can help solve problems with circular references stopping objects
        from being garbage collected, and is a quick way to ensure that all
        memory associated to Brian objects is deleted.
        
    Notes
    -----
    
    Removes the object from `brian_objects`. Will also set the
    `BrianObject.active` flag to ``False`` for already existing `Network`
    objects. Calls a garbage collection on completion.
    
    See ALso
    --------
    
    run, reinit, MagicError
    '''
    brian_objects.state = brian_objects.STATE_VALID
    if erase:
        for obj in brian_objects:
            for k, v in obj.__dict__.iteritems():
                object.__setattr__(obj, k, None)
    brian_objects.clear()
    gc.collect()
