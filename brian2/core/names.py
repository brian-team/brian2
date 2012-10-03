from brian2.core.tracking import Trackable, InstanceTrackerSet

__all__ = ['Nameable']


class Nameable(Trackable):
    '''
    Base class to find a unique name for an object
    
    If you specify a name explicitly, and it has already been taken, a
    `ValueError` is raised. If a name is not specified, it will try names of
    the form ``basename_0``, ``basename_1`` until it finds one which hasn't
    been taken.
    
    Parameters
    ----------
    name : (str, None), optional
        An explicit name, if not specified gives an automatically generated name
        
    Raises
    ------
    ValueError
        If the name is already taken.
    '''
    #: Stem of automatically generated names
    basename = 'nameable_object'
    
    def _find_name(self, name):
        basename = self.basename
        allnames = set(obj().name for obj in Nameable.__instances__() if hasattr(obj(), 'name'))
        if name is not None:
            if name in allnames:
                raise ValueError("An object with name "+name+" is already defined.")
            return name
        i = 0
        while basename+'_'+str(i) in allnames:
            i += 1
        return basename+'_'+str(i)
    
    def __init__(self, name=None):
        self._name = self._find_name(name)

    name = property(fget=lambda self:self._name,
                    doc='''
                        The unique name for this object.
                        
                        Used when generating code. Should be an acceptable
                        variable name, i.e. starting with a letter
                        character and followed by alphanumeric characters and
                        ``_``.
                        ''')

    
if __name__=='__main__':
    from brian2 import *
    from brian2.core.names import Nameable
    nam = Nameable()
    obj = BrianObject()
    obj2 = BrianObject()
    print nam.name, obj.name, obj2.name

