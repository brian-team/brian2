'''
Module containing the `Device` base class as well as the `RuntimeDevice`
implementation and some helper functions to access/set devices.
'''

import numpy as np

from brian2.memory.dynamicarray import DynamicArray, DynamicArray1D
from brian2.codegen.codeobject import create_codeobject
from brian2.codegen.targets import codegen_targets
from brian2.core.preferences import brian_prefs
from brian2.core.variables import ArrayVariable, DynamicArrayVariable
from brian2.units.fundamentalunits import Unit
from brian2.utils.logger import get_logger

__all__ = ['Device', 'RuntimeDevice',
           'get_device', 'set_device',
           'all_devices',
           'insert_device_code',
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
    
    def array(self, owner, name, size, unit, value=None, dtype=None, constant=False,
              is_bool=False, read_only=False):
        raise NotImplementedError()

    def arange(self, owner, name, size, start=0, dtype=None, constant=True,
               read_only=True):
        raise NotImplementedError()

    def dynamic_array_1d(self, owner, name, size, unit, dtype=None,
                         constant=False, constant_size=True, is_bool=False,
                         read_only=False):
        raise NotImplementedError()

    def dynamic_array(self, owner, name, size, unit, dtype=None,
                      constant=False, constant_size=True, is_bool=False,
                      read_only=False):
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

    def code_object(self, owner, name, abstract_code, namespace, variables, template_name,
                    variable_indices, codeobj_class=None,
                    template_kwds=None):
        codeobj_class = self.code_object_class(codeobj_class)
        return create_codeobject(owner, name, abstract_code, namespace, variables, template_name,
                                 variable_indices=variable_indices, codeobj_class=codeobj_class,
                                 template_kwds=template_kwds)
    
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
    
    
class RuntimeDevice(Device):
    '''
    '''
    def __init__(self):
        super(Device, self).__init__()

    def array(self, owner, name, size, unit, value=None, dtype=None,
              constant=False, is_bool=False, read_only=False):
        if is_bool:
            dtype = np.bool
        elif value is not None:
            dtype = value.dtype
        elif dtype is None:
            dtype = brian_prefs['core.default_scalar_dtype']
        if value is None:
            value = np.zeros(size, dtype=dtype)
        return ArrayVariable(name, unit, value, group_name=owner.name,
                             constant=constant, is_bool=is_bool,
                             read_only=read_only)

    def arange(self, owner, name, size, start=0, dtype=np.int32, constant=True,
               read_only=True):
        array = np.arange(start=start, stop=start+size, dtype=dtype)
        return ArrayVariable(name, Unit(1), array, group_name=owner.name,
                             constant=constant, is_bool=False,
                             read_only=read_only)

    def dynamic_array_1d(self, owner, name, size, unit, dtype=None,
                         constant=False,constant_size=True, is_bool=False,
                         read_only=False):
        if is_bool:
            dtype = np.bool
        if dtype is None:
            dtype = brian_prefs['core.default_scalar_dtype']
        array = DynamicArray1D(size, dtype=dtype)
        return DynamicArrayVariable(name, unit, array, dimensions=1,
                                    group_name=owner.name,
                                    constant=constant,
                                    constant_size=constant_size,
                                    is_bool=is_bool,
                                    read_only=read_only)

    def dynamic_array(self, owner, name, size, unit, dtype=None,
                      constant=False, constant_size=True, is_bool=False,
                      read_only=False):
        if is_bool:
            dtype = np.bool
        if dtype is None:
            dtype = brian_prefs['core.default_scalar_dtype']
        array = DynamicArray(size, dtype=dtype)
        return DynamicArrayVariable(name, unit, array, dimensions=len(size),
                                    group_name=owner.name,
                                    constant=constant,
                                    constant_size=constant_size,
                                    is_bool=is_bool,
                                    read_only=read_only)

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

current_device = runtime_device

def get_device():
    '''
    Gets the current `Device` object
    '''
    global current_device
    return current_device

def set_device(device):
    '''
    Sets the current `Device` object
    '''
    global current_device
    if isinstance(device, str):
        device = all_devices[device]
    current_device = device
    current_device.activate()


def insert_device_code(slot, code):
    '''
    Inserts the given set of code into the slot defined by the device.
    
    The behaviour of this function is device dependent. The runtime device ignores it (useful for debugging).
    '''
    get_device().insert_device_code(slot, code)
    