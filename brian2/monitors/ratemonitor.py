import weakref
from collections import defaultdict

import numpy as np

from brian2.codegen.codeobject import create_codeobject
from brian2.core.base import BrianObject
from brian2.core.preferences import brian_prefs
from brian2.core.scheduler import Scheduler
from brian2.core.variables import Variable, AttributeVariable
from brian2.memory.dynamicarray import DynamicArray1D
from brian2.units.allunits import second, hertz
from brian2.units.fundamentalunits import Unit, Quantity

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

        # create data structures
        self.reinit()

        self.variables = {'t': AttributeVariable(second, self.clock, 't'),
                           'dt': AttributeVariable(second, self.clock,
                                                   'dt', constant=True),
                           '_spikes': AttributeVariable(Unit(1),
                                                        self.source, 'spikes'),
                           # The template needs to have access to the
                           # DynamicArray here, having access to the underlying
                           # array is not enough since we want to do the resize
                           # in the template
                           '_rate': Variable(Unit(1), self._rate),
                           '_t': Variable(Unit(1), self._t),
                           '_num_source_neurons': Variable(Unit(1),
                                                           len(self.source))}

    def reinit(self):
        '''
        Clears all recorded rates
        '''
        self._rate = DynamicArray1D(0, use_numpy_resize=True,
                                    dtype=brian_prefs['core.default_scalar_dtype'])
        self._t = DynamicArray1D(0, use_numpy_resize=True,
                                 dtype=getattr(self.clock.t, 'dtype',
                                               np.dtype(type(self.clock.t))))

    def pre_run(self, namespace):
        self.codeobj = create_codeobject(self.name,
                                         '', # No model-specific code
                                         {}, # no namespace
                                         self.variables,
                                         template_name='ratemonitor',
                                         indices={},
                                         variable_indices=defaultdict(lambda: '_element'))

    def update(self):
        self.codeobj()

    @property
    def rate(self):
        '''
        Array of recorded rates (in units of Hz).
        '''
        return Quantity(self._rate.data.copy(), dim=hertz.dim)

    @property
    def rate_(self):
        '''
        Array of recorded rates (unitless).
        '''
        return self._rate.data.copy()

    @property
    def t(self):
        '''
        Array of recorded time points (in units of second).
        '''
        return Quantity(self._t.data.copy(), dim=second.dim)

    @property
    def t_(self):
        '''
        Array of recorded time points (unitless).
        '''
        return self._t.data.copy()

    def __repr__(self):
        description = '<{classname}, recording {source}>'
        return description.format(classname=self.__class__.__name__,
                                  source=self.source.name)
