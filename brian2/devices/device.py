import numpy as np

from brian2.memory.dynamicarray import DynamicArray, DynamicArray1D
from brian2.codegen.codeobject import create_codeobject
from brian2.codegen.targets import codegen_targets
from brian2.core.preferences import brian_prefs
from brian2.core.variables import ArrayVariable, DynamicArrayVariable

__all__ = ['Device', 'RuntimeDevice',
           'get_device', 'set_device',
           'all_devices',
           ]

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
    
    def array(self, owner, name, size, unit, dtype=None, constant=False,
              is_bool=False):
        raise NotImplementedError()

    def dynamic_array_1d(self, owner, name, size, unit, dtype=None,
                         constant=False, constant_size=True, is_bool=False):
        raise NotImplementedError()

    def dynamic_array(self, owner, name, size, unit, dtype=None,
                      constant=False, constant_size=True, is_bool=False):
        raise NotImplementedError()

    def code_object_class(self, codeobj_class=None):
        if codeobj_class is None:
            codeobj_class = get_default_codeobject_class()
        return codeobj_class

    def code_object(self, owner, name, abstract_code, namespace, variables, template_name,
                    indices, variable_indices, codeobj_class=None,
                    template_kwds=None):
        codeobj_class = self.code_object_class(codeobj_class)
        return create_codeobject(owner, name, abstract_code, namespace, variables, template_name,
                                 indices, variable_indices, codeobj_class=codeobj_class,
                                 template_kwds=template_kwds)
    
    def activate(self):
        '''
        Called when this device is set as the current device.
        '''
        pass

    
class RuntimeDevice(Device):
    '''
    '''
    def __init__(self):
        super(Device, self).__init__()

    def array(self, owner, name, size, unit, dtype=None,
              constant=False, is_bool=False):
        if is_bool:
            dtype = np.bool
        elif dtype is None:
            dtype = brian_prefs['core.default_scalar_dtype']
        array = np.zeros(size, dtype=dtype)
        return ArrayVariable(name, unit, array, group_name=owner.name,
                             constant=constant, is_bool=is_bool)

    def dynamic_array_1d(self, owner, name, size, unit, dtype=None,
                         constant=False,constant_size=True, is_bool=False):
        if is_bool:
            dtype = np.bool
        if dtype is None:
            dtype = brian_prefs['core.default_scalar_dtype']
        array = DynamicArray1D(size, dtype=dtype)
        return DynamicArrayVariable(name, unit, array, group_name=owner.name,
                                    constant=constant,
                                    constant_size=constant_size,
                                    is_bool=is_bool)

    def dynamic_array(self, owner, name, size, unit, dtype=None,
                      constant=False, constant_size=True, is_bool=False):
        if is_bool:
            dtype = np.bool
        if dtype is None:
            dtype = brian_prefs['core.default_scalar_dtype']
        array = DynamicArray(size, dtype=dtype)
        return DynamicArrayVariable(name, unit, array, group_name=owner.name,
                                    constant=constant,
                                    constant_size=constant_size,
                                    is_bool=is_bool)


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
