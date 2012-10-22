import weakref

from brian2 import BrianObject, brian_prefs, second, Scheduler
from brian2.memory.dynamicarray import DynamicArray1D

__all__ = ['SpikeMonitor']

class SpikeMonitor(BrianObject):
    '''
    Record spikes from a `NeuronGroup` or other spike source
    
    Parameters
    ----------
    source : (`NeuronGroup`, spike source)
        The source of spikes to record.
    when : `Scheduler`, optional
        When to record the spikes, by default uses the clock of the source
        and records spikes in the slot 'end'.
    name : str, optional
        A unique name for the object, otherwise will use ``spikemonitor_0``,
        etc.
    '''
    basename = 'spikemonitor'
    def __init__(self, source, when=None, name=None):
        scheduler = Scheduler(when)
        if not scheduler.defined_clock:
            scheduler.clock = source.clock
        if not scheduler.defined_when:
            scheduler.when = 'end'
        BrianObject.__init__(self, when=scheduler, name=name)
        self.source = weakref.proxy(source)
        self.reinit()
        
    def reinit(self):
        '''
        Clears all recorded spikes
        '''
        self._i = DynamicArray1D(0, use_numpy_resize=True, dtype=int)
        self._t = DynamicArray1D(0, use_numpy_resize=True,
                                 dtype=brian_prefs.default_scalar_dtype)
        
    def update(self):
        spikes = self.source.spikes
        nspikes = len(spikes)
        if nspikes:
            i = self._i
            t = self._t
            oldsize = len(i)
            newsize = oldsize+nspikes
            i.resize(newsize)
            t.resize(newsize)
            i.data[oldsize:] = spikes
            t.data[oldsize:] = self.clock.t_
            
    @property
    def i(self):
        '''
        Array of recorded spike indices, with corresponding times `t`.
        '''
        return self._i.data.copy()
    
    @property
    def t(self):
        '''
        Array of recorded spike times, with corresponding indices `i`.
        '''
        return self._t.data.copy()*second

    @property
    def t_(self):
        '''
        Array of recorded spike times without units, with corresponding indices `i`.
        '''
        return self._t.data.copy()
    
    @property
    def it(self):
        '''
        Returns the pair (`i`, `t`).
        '''
        return self.i, self.t

    @property
    def it_(self):
        '''
        Returns the pair (`i`, `t_`).
        '''
        return self.i, self.t_
    
    @property
    def num_spikes(self):
        '''
        Returns the number of recorded spikes
        '''
        return len(self.i)  

    
if __name__=='__main__':
    from pylab import *
    from brian2 import *
    from brian2.codegen.languages import *
    import time

    N = 100
    tau = 10*ms
    eqs = '''
    dV/dt = (2-V)/tau : 1
    '''
    threshold = 'V>1'
    reset = 'V = 0'
    G = NeuronGroup(N, eqs, threshold=threshold, reset=reset)
    G.V = rand(N)
    M = SpikeMonitor(G)
    run(100*ms)
    print "Recorded", M.num_spikes, "spikes"
    i, t = M.it
    print t
    plot(t, i, '.k')
    show()
