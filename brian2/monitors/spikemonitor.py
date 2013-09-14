import weakref
from collections import defaultdict

import numpy as np

from brian2.codegen.codeobject import create_codeobject
from brian2.core.base import BrianObject
from brian2.core.preferences import brian_prefs
from brian2.core.scheduler import Scheduler
from brian2.core.variables import ArrayVariable, AttributeVariable, Variable, DynamicArrayVariable
from brian2.units.allunits import second
from brian2.units.fundamentalunits import Unit
from brian2.devices.device import get_device

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

        # Handle subgroups correctly
        start = getattr(self.source, 'start', 0)
        end = getattr(self.source, 'end', len(self.source))

        self.variables = {'t': AttributeVariable(second, self.clock, 't'),
                          '_spikespace': self.source.variables['_spikespace'],
                           '_i': DynamicArrayVariable('_i', Unit(1), self._i, group_name=self.name),
                           '_t': DynamicArrayVariable('_t', Unit(1), self._t, group_name=self.name),
                           '_count': ArrayVariable('_count', Unit(1), self.count, group_name=self.name),
                           '_source_start': Variable(Unit(1), start,
                                                     constant=True),
                           '_source_end': Variable(Unit(1), end,
                                                   constant=True)}

    def reinit(self):
        '''
        Clears all recorded spikes
        '''
        dev = get_device()
        self._i = dev.dynamic_array_1d(self, '_i', 0, 1, dtype=np.int32)
        self._t = dev.dynamic_array_1d(self, '_t', 0, 1, dtype=brian_prefs['core.default_scalar_dtype'])
        
        #: Array of the number of times each source neuron has spiked
        self.count = get_device().array(self, '_count', len(self.source), 1, dtype=np.int32)

    def pre_run(self, namespace):
        self.codeobj = get_device().code_object(
                                         self,
                                         self.name+'_codeobject*',
                                         '', # No model-specific code
                                         {}, # no namespace
                                         self.variables,
                                         template_name='spikemonitor',
                                         indices={},
                                         variable_indices=defaultdict(lambda: '_idx'),
                                         codeobj_class=self.codeobj_class)
        self.code_objects[:] = [weakref.proxy(self.codeobj)]
        self.updaters[:] = [self.codeobj.get_updater()]

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
