'''
Brian data type

Helper for handling Variable objects in code generation
'''

from brian2.core.variables import (Variable, Constant, AttributeVariable,
                                   ArrayVariable, DynamicArrayVariable,
                                   Subexpression, AuxiliaryVariable,
                                   )
from brian2.memory.dynamicarray import DynamicArray

import numpy as np


__all__ = ['BrianDataType']


def get_dtype(val):
    if isinstance(val, type):
        return get_dtype(val())

    is_bool = (val is True or
               val is False or
               val is np.True_ or
               val is np.False_)
    if is_bool:
        return 'bool'
    if hasattr(val, 'dtype'):
        return val.dtype.name
    if isinstance(val, int):
        return 'int'    
    if isinstance(val, float):
        return 'float'
    
    return 'unknown[%s, %s]' % (str(val), val.__class__.__name__)

def get_var_dtype(val):
    return get_dtype(val.dtype)


class BrianDataType(object):
    '''
    array_type : 'scalar', 'array', 'dynamic_array'
    data_type : 'bool', 'int', ...
    constant : bool
    
    Allowed data types:
    
    bool, int, float, int32, int64, uint32, uint64, float32, float64, ...
    '''
    def __init__(self, var):
        self.var = var
        if isinstance(var, Constant):
            self.array_type = 'scalar'
            self.data_type = get_dtype(var.value)
            self.constant = True
        elif isinstance(var, DynamicArrayVariable):
            self.array_type = 'dynamic_array'
            self.data_type = get_dtype(var.dtype)
            self.constant = var.constant            
        elif isinstance(var, ArrayVariable):
            if var.scalar:
                raise NotImplementedError
            self.array_type = 'array'
            self.data_type = get_dtype(var.dtype)
            self.constant = var.constant
        elif isinstance(var, AttributeVariable):
            val = getattr(var.obj, var.attribute)
            if var.scalar:
                self.array_type = 'scalar'
            elif isinstance(val, DynamicArray):
                self.array_type = 'dynamic_array'
            elif isinstance(val, (np.ndarray, list, tuple)):
                self.array_type = 'array'
            else:
                raise NotImplementedError
            self.constant = False
            self.data_type = get_dtype(val)
        else:
            raise NotImplementedError
        
        # doesn't handle Constant correctly
        #self.data_type = get_var_dtype(var)
    
    def __str__(self):
        s = '%s(%s)' % (self.array_type, self.data_type)
        if self.constant:
            s = 'constant.'+s
        return s
        
if __name__=='__main__':
    from brian2 import *
    print BrianDataType(Constant('x', volt, 3*mV))
    print BrianDataType(ArrayVariable('x', volt, None, 10, device))
    print BrianDataType(ArrayVariable('x', 1, None, 10, device, dtype=int))
    print BrianDataType(DynamicArrayVariable('x', 1, None, 10, device, dtype=bool))
    print BrianDataType(AttributeVariable('t', second, defaultclock, 't', None))
    