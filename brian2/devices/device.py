import numpy
from brian2.memory.dynamicarray import DynamicArray, DynamicArray1D
from brian2.codegen.codeobject import create_codeobject
from brian2.codegen.targets import codegen_targets
from brian2.core.preferences import brian_prefs

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
    
    def create_array(self, owner, name, size, unit, dtype=None):
        pass

    def create_dynamic_array_1d(self, owner, name, size, unit, dtype=None):
        pass

    def create_dynamic_array(self, owner, name, size, unit, dtype=None):
        pass
    
    def code_object_class(self, codeobj_class=None):
        if codeobj_class is None:
            codeobj_class = get_default_codeobject_class()
        return codeobj_class

    def code_object(self, name, abstract_code, namespace, variables, template_name,
                    indices, variable_indices, codeobj_class=None,
                    template_kwds=None):
        codeobj_class = self.code_object_class(codeobj_class)
        return create_codeobject(name, abstract_code, namespace, variables, template_name,
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
        pass

    def array(self, owner, name, size, unit, dtype=None):
        if dtype is None:
            dtype = brian_prefs['core.default_scalar_dtype']
        return numpy.zeros(size, dtype=dtype)

    def dynamic_array_1d(self, owner, name, size, unit, dtype):
        if dtype is None:
            dtype = brian_prefs['core.default_scalar_dtype']
        return DynamicArray1D(size, dtype=dtype)

    def dynamic_array(self, owner, name, size, unit, dtype):
        if dtype is None:
            dtype = brian_prefs['core.default_scalar_dtype']
        return DynamicArray(size, dtype=dtype)


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
