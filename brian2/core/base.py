'''
All Brian objects should derive from `BrianObject`.
'''

import gc

from brian2.utils.logger import get_logger
from brian2.core.scheduler import Scheduler
from brian2.core.names import Nameable

__all__ = ['BrianObject',
           'clear',
           ]

logger = get_logger(__name__)


class BrianObject(Nameable):
    '''
    All Brian objects derive from this class, defines magic tracking and update.

    See the documentation for `Network` for an explanation of which
    objects get updated in which order.
    
    Parameters
    ----------
    when : `Scheduler`, optional
        Defines when the object is updated in the main `Network.run`
        loop.
    name : str, optional
        A unique name for the object - one will be assigned automatically if
        not provided (of the form ``brianobject_1``, etc.) based on the
        stem `basename`.

    Notes
    -----
        
    The set of all `BrianObject` objects is stored in ``BrianObject.__instances__()``.
    
    Brian objects deriving from this class should always define an
    ``update()`` method, that gets called by `Network.run`.    
    '''
    def __init__(self, when=None, name=None):
        scheduler = Scheduler(when)
        when = scheduler.when
        order = scheduler.order
        clock = scheduler.clock
        
        Nameable.__init__(self, name)
     
        # : The ID string determining when the object should be updated in `Network.run`.
        self.when = when
        
        #: The order in which objects with the same clock and ``when`` should be updated
        self.order = order
        
#        #: The `Clock` determining when the object should be updated.
#        self.clock = clock
        self._clock = clock
        
        self._contained_objects = []
        
        self._active = True
        
        logger.debug("Created BrianObject with name {self.name}, "
                     "clock name {self.clock.name}, "
                     "when={self.when}, order={self.order}".format(self=self))

    #: The stem for the automatically generated `name` attribute
    basename = 'brianobject'

    #: Whether or not `MagicNetwork` is invalidated when a new `BrianObject` of this type is created or removed
    invalidates_magic_network = True
    
    def prepare(self):
        '''
        Optional method to prepare data for the first time.
        
        Called by `Network.prepare`. Note that this method will not be
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
        Reinitialise the object, called by `Network.reinit`.
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

    # This is a repeat from Nameable.name, but we want to get the documentation
    # here again
    name = Nameable.name
    
    
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
    
    Removes the objects from ``BrianObject.__instances__()`` and
    ``Nameable.__instances__()``.
    Will also set the
    `BrianObject.active` flag to ``False`` for already existing `Network`
    objects. Calls a garbage collection on completion.
    
    See ALso
    --------
    
    run, reinit, MagicError
    '''
    if erase:
        for obj in BrianObject.__instances__():
            obj = obj()
            for k, v in obj.__dict__.iteritems():
                object.__setattr__(obj, k, None)
    BrianObject.__instances__().clear()
    Nameable.__instances__().clear()
    gc.collect()
