from brian2.units.fundamentalunits import fail_for_dimension_mismatch

__all__ = ['Group']

class Group(object):
    '''
    Mix-in class for accessing arrays by attribute.
    
    # TODO: Overwrite the __dir__ method to return the state variables
    # (should make autocompletion work)
    '''
    def __init__(self):
        if not hasattr(self, 'specifiers'):
            raise ValueError('Classes derived from Group need specifiers attribute')
        self._group_attribute_access_active = True
    
    def state_(self, name):
        '''
        Gets the unitless array.
        '''
        try:
            return self.specifiers[name].get_value()
        except KeyError:
            raise KeyError("Array named "+name+" not found.")
        
    def state(self, name):
        '''
        Gets the array with units.
        '''
        try:
            spec = self.specifiers[name]
            # TODO: More efficitent to use Quantity.with_dimensions ?
            return spec.get_value() * spec.unit
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
        elif name in self.specifiers:
            spec = self.specifiers[name]
            fail_for_dimension_mismatch(val, spec.unit,
                                        'Incorrect units for setting %s' % name)
            spec.set_value(val)
        elif len(name) and name[-1]=='_' and name[:-1] in self.specifiers:
            # no unit checking
            self.specifiers[name[:-1]].set_value(val)
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
