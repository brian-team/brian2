'''
Module containing the `Device` base class as well as the `RuntimeDevice`
implementation and some helper functions to access/set devices.
'''
from weakref import WeakKeyDictionary

import numpy as np

from brian2.memory.dynamicarray import DynamicArray, DynamicArray1D
from brian2.codegen.targets import codegen_targets
from brian2.codegen.runtime.numpy_rt import NumpyCodeObject
from brian2.core.names import find_name
from brian2.core.preferences import brian_prefs
from brian2.core.variables import ArrayVariable, DynamicArrayVariable
from brian2.core.functions import Function
from brian2.utils.logger import get_logger
from brian2.utils.stringtools import code_representation, indent

__all__ = ['Device', 'RuntimeDevice',
           'get_device', 'set_device',
           'all_devices',
           'device_override',
           'device',
           ]

logger = get_logger(__name__)

all_devices = {}


def get_default_codeobject_class():
    '''
    Returns the default `CodeObject` class from the preferences.
    '''
    codeobj_class = brian_prefs['codegen.target']
    if isinstance(codeobj_class, str):
        for target in codegen_targets:
            if target.class_name == codeobj_class:
                return target
        # No target found
        raise ValueError("Unknown code generation target: %s, should be "
                         " one of %s"%(codeobj_class,
                                       [target.class_name
                                        for target in codegen_targets]))
    return codeobj_class


class Device(object):
    '''
    Base Device object.
    '''
    def __init__(self):
        pass

    def get_array_name(self, var, access_data=True):
        '''
        Return a globally unique name for `var`.

        Parameters
        ----------
        access_data : bool, optional
            For `DynamicArrayVariable` objects, specifying `True` here means the
            name for the underlying data is returned. If specifying `False`,
            the name of object itself is returned (e.g. to allow resizing).

        Returns
        -------
        name : str
            The name for `var`.
        '''
        raise NotImplementedError()

    def get_len(self, array):
        '''
        Return the length of the array.

        Parameters
        ----------
        array : `ArrayVariable`
            The array for which the length is requested.

        Returns
        -------
        l : int
            The length of the array.
        '''
        raise NotImplementedError()

    def add_array(self, var):
        '''
        Add an array to this device.

        Parameters
        ----------
        var : `ArrayVariable`
            The array to add.
        '''
        raise NotImplementedError()

    def init_with_zeros(self, var):
        '''
        Initialize an array with zeros.

        Parameters
        ----------
        var : `ArrayVariable`
            The array to initialize with zeros.
        '''
        raise NotImplementedError()

    def init_with_arange(self, var, start):
        '''
        Initialize an array with an integer range.

        Parameters
        ----------
        var : `ArrayVariable`
            The array to fill with the integer range.
        start : int
            The start value for the integer range
        '''
        raise NotImplementedError()

    def fill_with_array(self, var, arr):
        '''
        Fill an array with the values given in another array.

        Parameters
        ----------
        var : `ArrayVariable`
            The array to fill.
        arr : `ndarray`
            The array values that should be copied to `var`.
        '''
        raise NotImplementedError()

    def spike_queue(self, source_start, source_end):
        '''
        Create and return a new `SpikeQueue` for this `Device`.

        Parameters
        ----------
        source_start : int
            The start index of the source group (necessary for subgroups)
        source_end : int
            The end index of the source group (necessary for subgroups)
        '''
        raise NotImplementedError()

    def code_object_class(self, codeobj_class=None):
        if codeobj_class is None:
            codeobj_class = get_default_codeobject_class()
        return codeobj_class

    def code_object(self, owner, name, abstract_code, variables, template_name,
                    variable_indices, codeobj_class=None,
                    template_kwds=None, override_conditional_write=None):

        codeobj_class = self.code_object_class(codeobj_class)
        template = getattr(codeobj_class.templater, template_name)
        iterate_all = template.iterate_all
        generator = codeobj_class.generator_class(variables=variables,
                                                  variable_indices=variable_indices,
                                                  owner=owner,
                                                  iterate_all=iterate_all,
                                                  codeobj_class=codeobj_class,
                                                  override_conditional_write=override_conditional_write,
                                                  allows_scalar_write=template.allows_scalar_write)
        if template_kwds is None:
            template_kwds = dict()
        else:
            template_kwds = template_kwds.copy()

        # Check that all functions are available
        for varname, value in variables.iteritems():
            if isinstance(value, Function):
                try:
                    value.implementations[codeobj_class]
                except KeyError as ex:
                    # if we are dealing with numpy, add the default implementation
                    if codeobj_class is NumpyCodeObject:
                        value.implementations.add_numpy_implementation(value.pyfunc)
                    else:
                        raise NotImplementedError(('Cannot use function '
                                                   '%s: %s') % (varname, ex))

        logger.debug('%s abstract code:\n%s' % (name, indent(code_representation(abstract_code))))

        scalar_code, vector_code, kwds = generator.translate(abstract_code,
                                                             dtype=brian_prefs['core.default_float_dtype'])
        # Add the array names as keywords as well
        for varname, var in variables.iteritems():
            if isinstance(var, ArrayVariable):
                pointer_name = generator.get_array_name(var)
                template_kwds[varname] = pointer_name
                if hasattr(var, 'resize'):
                    dyn_array_name = generator.get_array_name(var,
                                                              access_data=False)
                    template_kwds['_dynamic_'+varname] = dyn_array_name


        template_kwds.update(kwds)
        logger.debug('%s snippet (scalar):\n%s' % (name, indent(code_representation(scalar_code))))
        logger.debug('%s snippet (vector):\n%s' % (name, indent(code_representation(vector_code))))

        name = find_name(name)

        code = template(scalar_code, vector_code,
                        owner=owner, variables=variables, codeobj_name=name,
                        variable_indices=variable_indices,
                        get_array_name=generator.get_array_name,
                        **template_kwds)
        logger.debug('%s code:\n%s' % (name, indent(code_representation(code))))

        codeobj = codeobj_class(owner, code, variables, name=name)
        codeobj.compile()
        return codeobj
    
    def activate(self):
        '''
        Called when this device is set as the current device.
        '''
        pass

    def insert_device_code(self, slot, code):
        '''
        Insert code directly into a given slot in the device. By default does nothing.
        '''
        logger.warn("Ignoring device code, unknown slot: %s, code: %s" % (slot, code))
        
    def build(self, **kwds):
        '''
        For standalone projects, called when the project is ready to be built. Does nothing for runtime mode.
        '''
        pass
    
    
class RuntimeDevice(Device):
    '''
    '''
    def __init__(self):
        super(Device, self).__init__()
        #: Mapping from `Variable` objects to numpy arrays (or `DynamicArray`
        #: objects). Arrays in this dictionary will disappear as soon as the
        #: last reference to the `Variable` object used as a key is gone
        self.arrays = WeakKeyDictionary()
        
    def get_array_name(self, var, access_data=True):
        # if no owner is set, this is a temporary object (e.g. the array
        # of indices when doing G.x[indices] = ...). The name is not
        # necessarily unique over several CodeObjects in this case.
        owner_name = getattr(var.owner, 'name', 'temporary')

        if isinstance(var, DynamicArrayVariable):
            if access_data:
                return '_array_' + owner_name + '_' + var.name
            else:
                return '_dynamic_array_' + owner_name + '_' + var.name
        elif isinstance(var, ArrayVariable):
            return '_array_' + owner_name + '_' + var.name
        else:
            raise TypeError(('Do not have a name for variable of type '
                             '%s') % type(var))

    def add_array(self, var):
        # This creates the actual numpy arrays (or DynamicArrayVariable objects)
        if isinstance(var, DynamicArrayVariable):
            if var.dimensions == 1:
                arr = DynamicArray1D(var.size, dtype=var.dtype)
            else:
                arr = DynamicArray(var.size, dtype=var.dtype)
        else:
            arr = np.empty(var.size, dtype=var.dtype)

        self.arrays[var] = arr

    def get_value(self, var, access_data=True):
        if isinstance(var, DynamicArrayVariable) and access_data:
                return self.arrays[var].data
        else:
            return self.arrays[var]

    def set_value(self, var, value):
        self.arrays[var][:] = value

    def resize(self, var, new_size):
        self.arrays[var].resize(new_size)

    def init_with_zeros(self, var):
        self.arrays[var][:] = 0

    def init_with_arange(self, var, start):
        self.arrays[var][:] = np.arange(start, stop=var.get_len()+start)

    def fill_with_array(self, var, arr):
        self.arrays[var][:] = arr

    def spike_queue(self, source_start, source_end):
        # Use the C++ version of the SpikeQueue when available
        try:
            from brian2.synapses.cythonspikequeue import SpikeQueue
            logger.info('Using the C++ SpikeQueue', once=True)
        except ImportError:
            from brian2.synapses.spikequeue import SpikeQueue
            logger.info('Using the Python SpikeQueue', once=True)

        return SpikeQueue(source_start=source_start, source_end=source_end)


runtime_device = RuntimeDevice()
all_devices['runtime'] = runtime_device

active_device = runtime_device


class Dummy(object):
    '''
    Dummy object
    '''
    def __getattr__(self, name):
        return Dummy()
    def __call__(self, *args, **kwds):
        return Dummy()
    def __enter__(self):
        return Dummy()
    def __exit__(self, type, value, traceback):
        pass
    def __getitem__(self, i):
        return Dummy()
    def __setitem__(self, i, val):
        pass
    
class CurrentDeviceProxy(object):
    '''
    Method proxy for access to the currently active device
    '''
    def __getattr__(self, name):
        if not hasattr(active_device, name):
            logger.warn("Active device does not have an attribute '%s', ignoring this" % name)
            attr = Dummy()
        else:
            attr = getattr(active_device, name)
        return attr

#: Proxy object to access methods of the current device
device = CurrentDeviceProxy()


def get_device():
    '''
    Gets the actve `Device` object
    '''
    global active_device
    return active_device

def set_device(device):
    '''
    Sets the active `Device` object
    '''
    global active_device
    if isinstance(device, str):
        device = all_devices[device]
    active_device = device
    active_device.activate()


def device_override(name):
    '''
    Decorates a function/method to allow it to be overridden by the current `Device`.
    
    The ``name`` is the function name in the `Device` to use as an override if it exists.
    '''
    def device_override_decorator(func):
        def device_override_decorated_function(*args, **kwds):
            curdev = get_device()
            if hasattr(curdev, name):
                return getattr(curdev, name)(*args, **kwds)
            else:
                return func(*args, **kwds)
        
        device_override_decorated_function.__doc__ = func.__doc__
        
        return device_override_decorated_function
    
    return device_override_decorator
