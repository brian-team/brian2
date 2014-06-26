'''
This is only a temporary bridge for using Brian 1 hears with Brian 2.

This will be removed as soon as brian2hears is working.

NOTES:
- Slicing sounds with Brian 2 units doesn't work, you need to either use Brian 1 units or replace calls to
  ``sound[:20*ms]`` with ``sound.slice(None, 20*ms)``, etc. 

TODO: FilterbankGroup

Not working examples:
- simple_anf
- sound_localisation_model
- sounds
- time_varying_filter1
'''

try:
    import brian as b1
    import brian.hears as b1h
except ImportError:
    raise ImportError("brian2.hears is a temporary bridge between Brian 2 and the version of Brian Hears from "
                      "Brian 1, you need to have Brian 1 installed to use it.")

from brian2.utils.logger import get_logger
from brian2.units.fundamentalunits import Quantity

from numpy import array, ndarray
from inspect import isclass, ismethod

logger = get_logger(__name__)

logger.warn("You are using the temporary bridge between Brian 2 and Brian Hears from Brian 1, this will be removed "
            "in a later version.")

def modify_arg(arg):
    '''
    Modify arguments to make them compatible with Brian 1.
    
    - Arrays of units are replaced with straight arrays
    - Single values are replaced with Brian 1 equivalents
    - Slices are handled so we can use e.g. sound[:20*ms]
    
    The second part was necessary because some functions/classes test if an object is an array or not to see if it
    is a sequence, but because brian2.Quantity derives from ndarray this was causing problems.
    '''
    if isinstance(arg, Quantity):
        if len(arg.shape)==0:
            arg = b1.Quantity.with_dimensions(arg, arg.dim._dims)
        else:
            arg = array(arg)
    elif isinstance(arg, slice):
        arg = slice(modify_arg(arg.start), modify_arg(arg.stop), modify_arg(arg.step))
    return arg

def wrap_units(f):
    '''
    Wrap a function to convert units into a form that Brian 1 can handle. Also, check the output argument, if it is
    a ``b1h.Sound`` wrap it.
    '''
    def new_f(*args, **kwds):
        newargs = []
        newkwds = {}
        for arg in args:
            newargs.append(modify_arg(arg))
        for k, v in kwds.items():
            newkwds[k] = modify_arg(v)
        rv = f(*newargs, **newkwds)
        if rv.__class__==b1h.Sound:
            rv.__class__ = BridgeSound
        return rv
    return new_f

def wrap_units_class(_C):
    '''
    Wrap a class to convert units into a form that Brian 1 can handle in all methods
    '''
    class new_class(_C):
        for _k in _C.__dict__.keys():
            _v = getattr(_C, _k)
            if hasattr(ndarray, _k) and getattr(ndarray, _k) is _v:
                continue
            if ismethod(_v):
                _v = wrap_units(_v)
                exec '%s = _v' % _k
        del _k
        del _v
    return new_class

WrappedSound = wrap_units_class(b1h.Sound)
class BridgeSound(WrappedSound):
    def slice(self, *args):
        return self.__getitem__(slice(*args))
Sound = BridgeSound

__all__ = [k for k in b1h.__dict__.keys() if not k.startswith('_')]
for k in __all__:
    if k=='Sound':
        continue
    curobj = getattr(b1h, k)
    if callable(curobj):
        if isclass(curobj):
            curobj = wrap_units_class(curobj)
        else:
            curobj = wrap_units(curobj)
    exec '%s = curobj' % k
