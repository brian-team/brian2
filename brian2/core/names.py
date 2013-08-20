from brian2.utils.logger import get_logger
from brian2.core.tracking import Trackable
import re

__all__ = ['Nameable']

logger = get_logger(__name__)


def find_name(name):
    if name.endswith('*'):
        name = name[:-1]
        wildcard = True
    else:
        wildcard = False
    instances = set(Nameable.__instances__())
    allnames = set(obj().name for obj in instances
                   if hasattr(obj(), 'name'))

    # Try the name without any additions first:
    if name not in allnames:
        return name
    elif not wildcard:
        raise ValueError("An object with name "+name+" is already defined.")

    # Name is already taken, try _1, _2, etc.
    i = 1
    while name+'_'+str(i) in allnames:
        i += 1
    return name+'_'+str(i)


class Nameable(Trackable):
    '''
    Base class to find a unique name for an object
    
    If you specify a name explicitly, and it has already been taken, a
    `ValueError` is raised. You can also specify a name with a wildcard asterisk
    in the end, i.e. in the form ``'name*'``. It will then try ``name`` first
    but if this is already specified, it will try ``name_1``, `name__2``, etc.
    This is the default mechanism used by most core objects in Brian, e.g.
    `NeuronGroup` uses a default name of ``'neurongroup*'``.
    
    Parameters
    ----------
    name : str
        An name for the object, possibly ending in ``*`` to specify that
        variants of this name should be tried if the name (without the asterisk)
        is already taken.
        
    Raises
    ------
    ValueError
        If the name is already taken.
    '''    
    def __init__(self, name):
        if not isinstance(name, basestring):
            raise TypeError(('"name" argument has to be a string, is type '
                             '{type} instead').format(type=repr(type(name))))
        if not re.match(r"[_A-Za-z][_a-zA-Z0-9]*\*?$", name):
            raise ValueError("Name %s not valid variable name" % name)

        self._name = find_name(name)
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
    nam = Nameable('nameable')
    obj = BrianObject(name='object*')
    obj2 = BrianObject(name='object*')
    print nam.name, obj.name, obj2.name
