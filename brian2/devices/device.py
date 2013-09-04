import numpy
from brian2.memory.dynamicarray import DynamicArray1D
from brian2.codegen.codeobject import create_codeobject
from brian2.core.preferences import brian_prefs

__all__ = ['Device', 'RuntimeDevice',
           'get_device', 'set_device',
           ]

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

    def code_object(self, name, abstract_code, namespace, variables, template_name,
                    indices, variable_indices, codeobj_class=None,
                    template_kwds=None):
        return create_codeobject(name, abstract_code, namespace, variables, template_name,
                                 indices, variable_indices, codeobj_class=codeobj_class,
                                 template_kwds=template_kwds)

    
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


runtime_device = RuntimeDevice()


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
    current_device = device
    
