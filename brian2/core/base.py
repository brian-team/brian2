'''
All Brian objects should derive from :class:`BrianObject`.
'''

from weakref import ref, proxy
import gc
import copy

import brian2.core.clocks as clocks
from brian2.core.scheduler import Scheduler

__all__ = ['BrianObject',
           'BrianObjectSet',
           'brian_objects',
           'clear',
           ]


class BrianObjectSet(set):
    '''
    A `set` of `weakref.ref` to all existing `BrianObject` objects.
    
    Should not normally be directly used, except internally by `MagicNetwork`
    and `clear`.
    '''
    def add(self, value):
        '''
        Adds a `weakref.ref` to the ``value``
        '''
        # The second argument to ref is a callback that is called with the
        # ref as argument when the object has been deleted, here we just
        # remove it from the set in that case
        wr = ref(value, self.remove)
        set.add(self, wr)
        
    def remove(self, value):
        '''
        Removes the ``value`` (which should be a weakref) if it is in the set
        
        Sometimes the value will have been removed from the set by `clear`,
        so we ignore `KeyError` in this case.
        '''
        try:
            set.remove(self, value)
        except KeyError:
            pass


#: 'A `BrianObjectSet` containing all instances of `BrianObject`
brian_objects = BrianObjectSet()


class BrianObject(object):
    '''
    All Brian objects derive from this class, defines magic tracking and update.

    See the documentation for `Network` for an explanation of which
    objects get updated in which order.
    
    Parameters
    ----------
    when : `Scheduler`, optional
        Defines when the object is updated in the main `Network.run`
        loop.

    Notes
    -----
        
    The set of all `BrianObject` objects is stored in `brian_objects`.
    
    Brian objects deriving from this class should always define an
    ``update()`` method, that gets called by `Network.run`.
    '''        
    def __init__(self, when=None):
        scheduler = Scheduler(when)
        when = scheduler.when
        order = scheduler.order
        clock = scheduler.clock
     
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
        pass
        
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
    if erase:
        for obj in brian_objects:
            obj = obj()
            for k, v in obj.__dict__.iteritems():
                object.__setattr__(obj, k, None)
    brian_objects.clear()
    gc.collect()
