'''
This is only a temporary bridge for using Brian 1 hears with Brian 2.

This will be removed as soon as brian2hears is working.

NOTES:

* Slicing sounds with Brian 2 units doesn't work, you need to either use Brian 1 units or replace calls to
  ``sound[:20*ms]`` with ``sound.slice(None, 20*ms)``, etc. 

TODO: handle properties (e.g. sound.duration)

Not working examples:

* time_varying_filter1 (care with units)
'''

try:
    import brian as b1
    import brian.hears as b1h
except ImportError:
    raise ImportError("brian2.hears is a temporary bridge between Brian 2 and the version of Brian Hears from "
                      "Brian 1, you need to have Brian 1 installed to use it.")

from brian2.core.clocks import Clock
from brian2.core.operations import network_operation
from brian2.groups.neurongroup import NeuronGroup
from brian2.utils.logger import get_logger
from brian2.units.fundamentalunits import Quantity
from brian2.units import second

from numpy import asarray, array, ndarray
from inspect import isclass, ismethod

logger = get_logger(__name__)

logger.warn("You are using the temporary bridge between Brian 2 and Brian Hears from Brian 1, this will be removed "
            "in a later version.")

def convert_unit_b1_to_b2(val):
    return Quantity.with_dimensions(float(val), arg.dim._dims)

def convert_unit_b2_to_b1(val):
    return b1.Quantity.with_dimensions(float(val), arg.dim._dims)


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
            arg = b1.Quantity.with_dimensions(float(arg), arg.dim._dims)
        else:
            arg = asarray(arg)
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
        elif isinstance(rv, b1.Quantity):
            rv = Quantity.with_dimensions(float(rv), rv.dim._dims)
        return rv
    return new_f

def wrap_units_property(p):
    fget = p.fget
    fset = p.fset
    fdel = p.fdel
    if fget is not None:
        fget = wrap_units(fget)
    if fset is not None:
        fset = wrap_units(fset)
    if fdel is not None:
        fdel = wrap_units(fdel)
    new_p = property(fget, fset, fdel)
    return new_p

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
            elif isinstance(_v, property):
                _v = wrap_units_property(_v)
                exec '%s = _v' % _k
        del _k
        del _v
    return new_class


WrappedSound = wrap_units_class(b1h.Sound)
class BridgeSound(WrappedSound):
    '''
    We add a new method slice because slicing with units can't work with Brian 2 units.
    '''
    def slice(self, *args):
        return self.__getitem__(slice(*args))
Sound = BridgeSound


class FilterbankGroup(NeuronGroup):
    def __init__(self, filterbank, targetvar, *args, **kwds):
        self.targetvar = targetvar
        self.filterbank = filterbank
        self.buffer = None
        filterbank.buffer_init()

        # Sanitize the clock - does it have the right dt value?
        if 'clock' in kwds:
            if int(1/kwds['clock'].dt)!=int(filterbank.samplerate):
                raise ValueError('Clock should have 1/dt=samplerate')
            kwds['clock'] = Clock(dt=float(kwds['clock'].dt)*second)
        else:
            kwds['clock'] = Clock(dt=1*second/float(filterbank.samplerate))        
        
        buffersize = kwds.pop('buffersize', 32)
        if not isinstance(buffersize, int):
            buffersize = int(buffersize*self.samplerate)
        self.buffersize = buffersize
        self.buffer_pointer = buffersize
        self.buffer_start = -buffersize
        
        NeuronGroup.__init__(self, filterbank.nchannels, *args, **kwds)
        
        @network_operation(clock=self.clock, when='start')
        def apply_filterbank_output():
            if self.buffer_pointer>=self.buffersize:
                self.buffer_pointer = 0
                self.buffer_start += self.buffersize
                self.buffer = self.filterbank.buffer_fetch(self.buffer_start, self.buffer_start+self.buffersize)
            setattr(self, targetvar, self.buffer[self.buffer_pointer, :])
            self.buffer_pointer += 1
        
        self.contained_objects.append(apply_filterbank_output)
        
    def reinit(self):
        NeuronGroup.reinit(self)
        self.filterbank.buffer_init()
        self.buffer_pointer = self.buffersize
        self.buffer_start = -self.buffersize

handled_explicitly = {'Sound', 'FilterbankGroup'}

__all__ = [k for k in b1h.__dict__.keys() if not k.startswith('_')]
for k in __all__:
    if k in handled_explicitly:
        continue
    curobj = getattr(b1h, k)
    if callable(curobj):
        if isclass(curobj):
            curobj = wrap_units_class(curobj)
        else:
            curobj = wrap_units(curobj)
    exec '%s = curobj' % k

__all__.extend(['convert_unit_b1_to_b2',
                'convert_unit_b2_to_b1',
                ])
