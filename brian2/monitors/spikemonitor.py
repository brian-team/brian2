import numpy as np

from brian2.core.base import BrianObject
from brian2.core.scheduler import Scheduler
from brian2.core.variables import Variables
from brian2.units.allunits import second
from brian2.units.fundamentalunits import Unit, Quantity
from brian2.groups.group import CodeRunner

__all__ = ['SpikeMonitor']


class SpikeMonitor(CodeRunner):
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
        self.record = bool(record)

        # run by default on source clock at the end
        scheduler = Scheduler(when)
        if not scheduler.defined_clock:
            scheduler.clock = source.clock
        if not scheduler.defined_when:
            scheduler.when = 'end'

        self.codeobj_class = codeobj_class
        CodeRunner.__init__(self, source, 'spikemonitor',
                            name=name, when=scheduler)

        # Handle subgroups correctly
        start = getattr(source, 'start', 0)
        stop = getattr(source, 'stop', len(source))

        self.variables = Variables(self)
        self.variables.add_clock_variables(scheduler.clock)
        self.variables.add_reference('_spikespace', source.variables['_spikespace'])
        self.variables.add_dynamic_array('_i', size=0, unit=Unit(1),
                                         dtype=np.int32, constant_size=False)
        self.variables.add_dynamic_array('_t', size=0, unit=Unit(1),
                                         constant_size=False)
        self.variables.add_array('_count', size=len(source), unit=Unit(1),
                                 dtype=np.int32)
        self.variables.add_constant('_source_start', Unit(1), start)
        self.variables.add_constant('_source_stop', Unit(1), stop)

    def reinit(self):
        '''
        Clears all recorded spikes
        '''
        raise NotImplementedError()

    @property
    def i(self):
        '''
        Array of recorded spike indices, with corresponding times `t`.
        '''
        return self.variables['_i'].get_value().copy()
    
    @property
    def t(self):
        '''
        Array of recorded spike times, with corresponding indices `i`.
        '''
        return Quantity(self.variables['_t'].get_value(), dim=second.dim,
                        copy=True)

    @property
    def t_(self):
        '''
        Array of recorded spike times without units, with corresponding indices `i`.
        '''
        return self.variables['_t'].get_value().copy()
    
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
    def count(self):
        '''
        Return the total spike count for each neuron.
        '''
        return self.variables['_count'].get_value().copy()

    @property
    def num_spikes(self):
        '''
        Returns the number of recorded spikes
        '''
        return sum(self.count)  

    def __repr__(self):
        description = '<{classname}, recording {source}>'
        return description.format(classname=self.__class__.__name__,
                                  source=self.group.name)
