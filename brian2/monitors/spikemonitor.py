import weakref

from numpy import zeros

from brian2 import BrianObject, brian_prefs, second, Scheduler
from brian2.memory.dynamicarray import DynamicArray1D

__all__ = ['SpikeMonitor']

class SpikeMonitor(BrianObject):
    '''
    Record spikes from a `NeuronGroup` or other spike source
    
    Parameters
    ----------
    source : (`NeuronGroup`, `SpikeSource`)
        The source of spikes to record.
    record : bool
        Whether or not to record each spike in `i` and `t` (the `count` will
        always be recorded).
    when : `Scheduler`, optional
        When to record the spikes, by default uses the clock of the source
        and records spikes in the slot 'end'.
    name : str, optional
        A unique name for the object, otherwise will use
        ``source.name+'_spikemonitor_0'``, etc.
    '''
    basename = 'spikemonitor'
    def __init__(self, source, record=True, when=None, name=None):
        self.source = weakref.proxy(source)
        self.record = bool(record)

        # run by default on source clock at the end
        scheduler = Scheduler(when)
        if not scheduler.defined_clock:
            scheduler.clock = source.clock
        if not scheduler.defined_when:
            scheduler.when = 'end'
        BrianObject.__init__(self, when=scheduler, name=name)
        
        # create data structures
        self.reinit()
        
    def reinit(self):
        '''
        Clears all recorded spikes
        '''
        self._i = DynamicArray1D(0, use_numpy_resize=True, dtype=int)
        self._t = DynamicArray1D(0, use_numpy_resize=True,
                                 dtype=brian_prefs.default_scalar_dtype)
        
        #: Array of the number of times each source neuron has spiked
        self.count = zeros(len(self.source), dtype=int)
        
    def update(self):
        spikes = self.source.spikes
        nspikes = len(spikes)
        if nspikes:
            if self.record:
                # update i, t arrays
                i = self._i
                t = self._t
                oldsize = len(i)
                newsize = oldsize+nspikes
                i.resize(newsize)
                t.resize(newsize)
                i.data[oldsize:] = spikes
                t.data[oldsize:] = self.clock.t_
            # update count
            self.count[spikes] += 1
            
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
        return sum(self.count)  

    
if __name__=='__main__':
    from pylab import *
    from brian2 import *
    from brian2.codegen.languages import *
    import time

    N = 100
    tau = 10*ms
    eqs = '''
    dV/dt = (2-V)/tau : 1
    Vt : 1
    '''
    threshold = 'V>Vt'
    reset = 'V = 0'
    G = NeuronGroup(N, eqs, threshold=threshold, reset=reset)
    G.Vt = arange(N)/(float(N))
    G.V = rand(N)*G.Vt
    M = SpikeMonitor(G)
    run(100*ms)
    print "Recorded", M.num_spikes, "spikes"
    i, t = M.it
    print t
    print M.name
    subplot(211)
    plot(t, i, '.k')
    subplot(212)
    plot(M.count)
    show()
