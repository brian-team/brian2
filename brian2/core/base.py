'''
All Brian objects should derive from `BrianObject`.
'''

import gc
import weakref

from brian2.utils.logger import get_logger
from brian2.core.names import Nameable
from brian2.core.clocks import Clock, defaultclock
from brian2.units.allunits import second
from brian2.units.fundamentalunits import check_units

__all__ = ['BrianObject',
           'weakproxy_with_fallback',
           ]

logger = get_logger(__name__)


class BrianObject(Nameable):
    '''
    All Brian objects derive from this class, defines magic tracking and update.

    See the documentation for `Network` for an explanation of which
    objects get updated in which order.
    
    Parameters
    ----------
    dt : `Quantity`, optional
        The time step to be used for the simulation. Cannot be combined with
        the `clock` argument.
    clock : `Clock`, optional
        The update clock to be used. If neither a clock, nor the `dt` argument
        is specified, the `defaultclock` will be used.
    when : str, optional
        In which scheduling slot to simulate the object during a time step.
        Defaults to ``'start'``.
    order : int, optional
        The priority of this object for operations occurring at the same time
        step and in the same scheduling slot. Defaults to 0.
    name : str, optional
        A unique name for the object - one will be assigned automatically if
        not provided (of the form ``brianobject_1``, etc.).

    Notes
    -----
        
    The set of all `BrianObject` objects is stored in ``BrianObject.__instances__()``.
    '''    
    @check_units(dt=second)
    def __init__(self, dt=None, clock=None, when='start', order=0, name='brianobject*'):

        if dt is not None and clock is not None:
            raise ValueError('Can only specify either a dt or a clock, not both.')

        if not isinstance(when, basestring):
            # Give some helpful error messages for users coming from the alpha
            # version
            if isinstance(when, Clock):
                raise TypeError(("Do not use the 'when' argument for "
                                 "specifying a clock, either provide a "
                                 "timestep for the 'dt' argument or a Clock "
                                 "object for 'clock'."))
            if isinstance(when, tuple):
                raise TypeError("Use the separate keyword arguments, 'dt' (or "
                                "'clock'), 'when', and 'order' instead of "
                                "providing a tuple for 'when'. Only use the "
                                "'when' argument for the scheduling slot.")
            # General error
            raise TypeError("The 'when' argument has to be a string "
                            "specifying the scheduling slot (e.g. 'start').")

        Nameable.__init__(self, name)

        #: The clock used for simulating this object
        self._clock = clock
        if clock is None:
            if dt is not None:
                self._clock = Clock(dt=dt, name=self.name+'_clock*')
            else:
                self._clock = defaultclock

        #: Used to remember the `Network` in which this object has been included
        #: before, to raise an error if it is included in a new `Network`
        self._network = None

        #: The ID string determining when the object should be updated in `Network.run`.
        self.when = when
        
        #: The order in which objects with the same clock and ``when`` should be updated
        self.order = order

        self._dependencies = set()
        self._contained_objects = []
        self._code_objects = []
        
        self._active = True
        
        #: The scope key is used to determine which objects are collected by magic
        self._scope_key = self._scope_current_key
        
        logger.debug("Created BrianObject with name {self.name}, "
                     "clock={self._clock}, "
                     "when={self.when}, order={self.order}".format(self=self))

    #: Global key value for ipython cell restrict magic
    _scope_current_key = 0
    
    #: Whether or not `MagicNetwork` is invalidated when a new `BrianObject` of this type is added
    invalidates_magic_network = True

    #: Whether or not the object should be added to a `MagicNetwork`. Note that
    #: all objects in `BrianObject.contained_objects` are automatically added
    #: when the parent object is added, therefore e.g. `NeuronGroup` should set
    #: `add_to_magic_network` to ``True``, but it should not be set for all the
    #: dependent objects such as `StateUpdater`
    add_to_magic_network = False

    def add_dependency(self, obj):
        '''
        Add an object to the list of dependencies. Takes care of handling
        subgroups correctly (i.e., adds its parent object).

        Parameters
        ----------
        obj : `BrianObject`
            The object that this object depends on.
        '''
        from brian2.groups.subgroup import Subgroup
        if isinstance(obj, Subgroup):
            self._dependencies.add(obj.source.id)
        else:
            self._dependencies.add(obj.id)

    def before_run(self, run_namespace=None, level=0):
        '''
        Optional method to prepare the object before a run.

        TODO
        '''
        pass
    
    def after_run(self):
        '''
        Optional method to do work after a run is finished.
        
        Called by `Network.after_run` after the main simulation loop terminated.
        '''
        pass

    def run(self):
        for codeobj in self._code_objects:
            codeobj()

    contained_objects = property(fget=lambda self:self._contained_objects,
                                 doc='''
         The list of objects contained within the `BrianObject`.
         
         When a `BrianObject` is added to a `Network`, its contained objects will
         be added as well. This allows for compound objects which contain
         a mini-network structure.
         
         Note that this attribute cannot be set directly, you need to modify
         the underlying list, e.g. ``obj.contained_objects.extend([A, B])``.
         ''')

    code_objects = property(fget=lambda self:self._code_objects,
                                 doc='''
         The list of `CodeObject` contained within the `BrianObject`.
         
         TODO: more details.
                  
         Note that this attribute cannot be set directly, you need to modify
         the underlying list, e.g. ``obj.code_objects.extend([A, B])``.
         ''')

    updaters = property(fget=lambda self:self._updaters,
                                 doc='''
         The list of `Updater` that define the runtime behaviour of this object.
         
         TODO: more details.
                  
         Note that this attribute cannot be set directly, you need to modify
         the underlying list, e.g. ``obj.updaters.extend([A, B])``.
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

    def __repr__(self):
        description = ('{classname}(clock={clock}, when={when}, order={order}, name={name})')
        return description.format(classname=self.__class__.__name__,
                                  when=self.when,
                                  clock=self._clock,
                                  order=self.order,
                                  name=repr(self.name))

    # This is a repeat from Nameable.name, but we want to get the documentation
    # here again
    name = Nameable.name


def weakproxy_with_fallback(obj):
    '''
    Attempts to create a `weakproxy` to the object, but falls back to the object if not possible.
    '''
    try:
        return weakref.proxy(obj)
    except TypeError:
        return obj

def device_override(name):
    '''
    Decorates a function/method to allow it to be overridden by the current `Device`.

    The ``name`` is the function name in the `Device` to use as an override if it exists.
    
    The returned function has an additional attribute ``original_function``
    which is a reference to the original, undecorated function.
    '''
    def device_override_decorator(func):
        def device_override_decorated_function(*args, **kwds):
            from brian2.devices.device import get_device
            curdev = get_device()
            if hasattr(curdev, name):
                return getattr(curdev, name)(*args, **kwds)
            else:
                return func(*args, **kwds)

        device_override_decorated_function.__doc__ = func.__doc__
        device_override_decorated_function.original_function = func

        return device_override_decorated_function

    return device_override_decorator
