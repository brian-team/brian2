import weakref
from collections import defaultdict

import numpy as np

from brian2.core.base import BrianObject
from brian2.core.scheduler import Scheduler
from brian2.core.variables import AttributeVariable, Variable
from brian2.units.allunits import second
from brian2.units.fundamentalunits import Unit, Quantity
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

        # Handle subgroups correctly
        start = getattr(self.source, 'start', 0)
        stop = getattr(self.source, 'stop', len(self.source))

        device = get_device()
        self.variables = {'t': AttributeVariable(second, self.clock, 't_'),
                          '_spikespace': self.source.variables['_spikespace'],
                           '_i': device.dynamic_array_1d(self, '_i', 0, Unit(1),
                                                         dtype=np.int32,
                                                         constant_size=False),
                           '_t': device.dynamic_array_1d(self, '_t', 0,
                                                         Unit(1),
                                                         constant_size=False),
                           '_count': device.array(self, '_count',
                                                  len(self.source),
                                                  Unit(1),
                                                  dtype=np.int32),
                           '_source_start': Variable(Unit(1), start,
                                                     constant=True),
                           '_source_stop': Variable(Unit(1), stop,
                                                   constant=True)}

    def reinit(self):
        '''
        Clears all recorded spikes
        '''
        raise NotImplementedError()

    def before_run(self, namespace):
        self.codeobj = get_device().code_object(
                                         self,
                                         self.name+'_codeobject*',
                                         '', # No model-specific code
                                         {}, # no namespace
                                         self.variables,
                                         template_name='spikemonitor',
                                         variable_indices=defaultdict(lambda: '_idx'),
                                         codeobj_class=self.codeobj_class)
        self.code_objects[:] = [weakref.proxy(self.codeobj)]
        self.updaters[:] = [self.codeobj.get_updater()]

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
                                  source=self.source.name)
