__all__ = ['Group']

class Group(object):
    '''
    Mix-in class for accessing arrays by attribute
    
    Defining object should create attributes `arrays`, `units` for fast look-up,
    and a more generic mechanism is to provide methods `get_array_`, `get_array`,
    `set_array_` and `set_array`. Call the `__init__` method once these have
    been successfully created (typically at the end of your `__init__` method).
    
    TODO: could this all be done more efficiently by creating properties at runtime?
    '''
    def __init__(self):
        if not hasattr(self, 'arrays') or not hasattr(self, 'units'):
            raise ValueError('Classes derived from Group need attributes arrays and units')
        self._group_attribute_access_active = True
        
    def get_array(self, name):
        raise KeyError
    get_array_ = get_array
    
    def set_array(self, name, val):
        raise KeyError
    set_array_ = set_array
    
    def state_(self, name):
        '''
        Gets the unitless array.
        '''
        if name in self.arrays:
            return self.arrays[name]
        try:
            return self.get_array_(name)
        except KeyError:
            raise KeyError("Array named "+name+" not found.")
        
    def state(self, name):
        '''
        Gets the array with units.
        '''
        if name in self.arrays:
            return self.arrays[name]*self.units[name]
        try:
            return self.get_array(name)
        except KeyError:
            raise KeyError("Array named "+name+" not found.")

    def __getattr__(self, name):
        # We do this because __setattr__ and __getattr__ are not active until
        # _group_attribute_access_active attribute is set, and if it is set,
        # then __getattr__ will not be called. Therefore, if getattr is called
        # with this name, it is because it hasn't been set yet and so this
        # method should raise an AttributeError to agree that it hasn't been
        # called yet.
        if name=='_group_attribute_access_active':
            raise AttributeError
        if not hasattr(self, '_group_attribute_access_active'):
            raise AttributeError
        # First try the version with units, and if it fails try the version
        # without units
        try:
            return self.state(name)
        except KeyError:
            if len(name) and name[-1]=='_':
                try:
                    origname = name[:-1]
                    return self.state_(origname)
                except KeyError:
                    raise AttributeError
            raise AttributeError

    def __setattr__(self, name, val):
        # attribute access is switched off until this attribute is created by
        # Group.__init__
        if not hasattr(self, '_group_attribute_access_active'):
            object.__setattr__(self, name, val)
        elif name in self.arrays:
            # TODO: unit checking
            self.arrays[name][:] = val
        elif len(name) and name[-1]=='_' and name[:-1] in self.arrays:
            # no unit checking
            self.arrays[name[:-1]][:] = val
        else:
            try:
                self.set_array(name, val)
            except KeyError:
                if name[-1]=='_':
                    try:
                        self.set_array_(name, val)
                    except KeyError:
                        object.__setattr__(self, name, val)
                else:
                    object.__setattr__(self, name, val)

    
if __name__=='__main__':
    from numpy import *
    from brian2 import *
    class TestGroup(Group):
        def __init__(self):
            self.arrays = {'x':ones(10),
                           }
            self.units = {'x':volt,
                          }
            Group.__init__(self)
    tg = TestGroup()
    tg.x_ = 5
    tg.y = 10
    print tg.x
    print tg.y
