from brian2.utils.logger import get_logger
from brian2.core.tracking import Trackable

__all__ = ['Nameable']

logger = get_logger(__name__)


class Nameable(Trackable):
    '''
    Base class to find a unique name for an object
    
    If you specify a name explicitly, and it has already been taken, a
    `ValueError` is raised. If a name is not specified, it will try names of
    the form ``basename_0``, ``basename_1`` until it finds one which hasn't
    been taken. If the object has a ``source`` attribute with a ``name``
    attribute, the base name will be given by ``source.name+'_'+basename``. If
    the object also has a ``target`` attribute the format will be
    ``sourcename_targetname_basename``. Note that to get this behaviour, the
    ``source`` and ``target`` attributes have to have been set at the time that
    ``Nameable.__init__`` is called.
    
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
        instances = Nameable.__instances__()
        allnames = set(obj().name for obj in instances if hasattr(obj(), 'name'))
        if name is not None:
            if name in allnames:
                raise ValueError("An object with name "+name+" is already defined.")
            return name
        basename = self.basename
        if hasattr(self, 'target') and hasattr(self.target, 'name'):
            basename = self.target.name+'_'+basename
        if hasattr(self, 'source') and hasattr(self.source, 'name'):
            basename = self.source.name+'_'+basename
        i = 0
        while basename+'_'+str(i) in allnames:
            i += 1
        return basename+'_'+str(i)
    
    def __init__(self, name=None):
        self._name = self._find_name(name)
        logger.debug("Created object of class "+self.__class__.__name__+" with name "+self._name)

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
