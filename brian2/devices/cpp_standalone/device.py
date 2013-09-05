import numpy
from brian2.devices.device import Device, set_device
from brian2.core.preferences import brian_prefs

class CPPStandaloneDevice(Device):
    '''
    '''
    def __init__(self):
        self.arrays = {}
        self.dynamic_arrays = {}
        
    def array(self, owner, name, size, unit, dtype=None):
        if dtype is None:
            dtype = brian_prefs['core.default_scalar_dtype']
        arr = numpy.zeros(size, dtype=dtype)
        self.arrays['_array_%s_%s' % (owner.name, name)] = arr
        return arr

    def dynamic_array_1d(self, owner, name, size, unit, dtype):
        if dtype is None:
            dtype = brian_prefs['core.default_scalar_dtype']
        arr = DynamicArray1D(size, dtype=dtype)
        self.dynamic_arrays['_array_%s_%s' % (owner.name, name)] = arr
        return arr


cpp_standalone_device = CPPStandaloneDevice()

set_device(cpp_standalone_device)
