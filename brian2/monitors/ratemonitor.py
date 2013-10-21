import weakref
from collections import defaultdict

import numpy as np

from brian2.core.base import BrianObject
from brian2.core.scheduler import Scheduler
from brian2.core.variables import (Variable, AttributeVariable)
from brian2.units.allunits import second, hertz
from brian2.units.fundamentalunits import Unit, Quantity
from brian2.devices.device import get_device

__all__ = ['PopulationRateMonitor']


class PopulationRateMonitor(BrianObject):
    '''
    Record instantaneous firing rates, averaged across neurons from a
    `NeuronGroup` or other spike source.

    Parameters
    ----------
    source : (`NeuronGroup`, `SpikeSource`)
        The source of spikes to record.
    when : `Scheduler`, optional
        When to record the spikes, by default uses the clock of the source
        and records spikes in the slot 'end'.
    name : str, optional
        A unique name for the object, otherwise will use
        ``source.name+'_ratemonitor_0'``, etc.
    codeobj_class : class, optional
        The `CodeObject` class to run code with.
    '''
    def __init__(self, source, when=None, name='ratemonitor*',
                 codeobj_class=None):
        self.source = weakref.proxy(source)

        # run by default on source clock at the end
        scheduler = Scheduler(when)
        if not scheduler.defined_clock:
            scheduler.clock = source.clock
        if not scheduler.defined_when:
            scheduler.when = 'end'

        self.codeobj_class = codeobj_class
        BrianObject.__init__(self, when=scheduler, name=name)

        dev = get_device()
        self.variables = {'t': AttributeVariable(second, self.clock, 't_'),
                          'dt': AttributeVariable(second, self.clock,
                                                  'dt_', constant=True),
                          '_spikespace': self.source.variables['_spikespace'],
                          '_rate': dev.dynamic_array_1d(self, '_rate', 0, 1,
                                                        constant_size=False),
                          '_t': dev.dynamic_array_1d(self, '_t', 0, second,
                                                     dtype=getattr(self.clock.t, 'dtype',
                                                                   np.dtype(type(self.clock.t))),
                                                     constant_size=False),
                          '_num_source_neurons': Variable(Unit(1),
                                                          len(self.source))}

    def reinit(self):
        '''
        Clears all recorded rates
        '''
        raise NotImplementedError()

    def before_run(self, namespace):
        self.codeobj = get_device().code_object(
                                         self,
                                         self.name+'_codeobject*',
                                         '', # No model-specific code
                                         {}, # no namespace
                                         self.variables,
                                         template_name='ratemonitor',
                                         variable_indices=defaultdict(lambda: '_idx'))

        self.updaters[:] = [self.codeobj.get_updater()]

    @property
    def rate(self):
        '''
        Array of recorded rates (in units of Hz).
        '''
        return Quantity(self.variables['_rate'].get_value(), dim=hertz.dim,
                        copy=True)

    @property
    def rate_(self):
        '''
        Array of recorded rates (unitless).
        '''
        return self.variables['_rate'].get_value().copy()

    @property
    def t(self):
        '''
        Array of recorded time points (in units of second).
        '''
        return Quantity(self.variables['_t'].get_value(), dim=second.dim,
                        copy=True)

    @property
    def t_(self):
        '''
        Array of recorded time points (unitless).
        '''
        return self.variables['_t'].get_value().copy()

    def __repr__(self):
        description = '<{classname}, recording {source}>'
        return description.format(classname=self.__class__.__name__,
                                  source=self.source.name)
