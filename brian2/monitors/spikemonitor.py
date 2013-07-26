import weakref

import numpy as np

from brian2.codegen.codeobject import create_codeobject
from brian2.core.base import BrianObject
from brian2.core.preferences import brian_prefs
from brian2.core.scheduler import Scheduler
from brian2.core.specifiers import ArrayVariable, AttributeVariable, Variable
from brian2.memory.dynamicarray import DynamicArray1D
from brian2.units.allunits import second
from brian2.units.fundamentalunits import Unit

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
    codeobj_class : class, optional
        The `CodeObject` class to run code with.
    '''
    def __init__(self, source, record=True, when=None, name='spikemonitor*',
                 codeobj_class=None):
        self.source = weakref.proxy(source)
        self.record = bool(record)

        # run by default on source clock at the end
        scheduler = Scheduler(when)
        if not scheduler.defined_clock:
            scheduler.clock = source.clock
        if not scheduler.defined_when:
            scheduler.when = 'end'

        self.codeobj_class = codeobj_class
        BrianObject.__init__(self, when=scheduler, name=name)
        
        # create data structures
        self.reinit()

        self.specifiers = {'t': AttributeVariable('t', second, self.clock, 't'),
                           '_spikes': AttributeVariable('_spikes', Unit(1),
                                                        self.source, 'spikes'),
                           # The template needs to have access to the
                           # DynamicArray here, having access to the underlying
                           # array is not enough since we want to do the resize
                           # in the template
                           '_i': Variable('_i', Unit(1), self._i),
                           '_t': Variable('_t', Unit(1), self._t),
                           '_count': ArrayVariable('_count', Unit(1),
                                                   self.count, ''),
                           '_num_source_neurons': Variable('_num_source_neurons',
                                                           Unit(1),
                                                           len(self.source))}

    def reinit(self):
        '''
        Clears all recorded spikes
        '''
        self._i = DynamicArray1D(0, use_numpy_resize=True, dtype=int)
        self._t = DynamicArray1D(0, use_numpy_resize=True,
                                 dtype=brian_prefs['core.default_scalar_dtype'])
        
        #: Array of the number of times each source neuron has spiked
        self.count = np.zeros(len(self.source), dtype=int)

    def pre_run(self, namespace):
        self.codeobj = create_codeobject(self.name,
                                         '', # No model-specific code
                                         {}, # no namespace
                                         self.specifiers,
                                         template_name='spikemonitor',
                                         indices={},
                                         iterate_all=[])

    def update(self):
        self.codeobj()
            
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

    def __repr__(self):
        description = '<{classname}, recording {source}>'
        return description.format(classname=self.__class__.__name__,
                                  source=self.source.name)

    
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
