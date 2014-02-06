import numpy as np

from brian2.core.scheduler import Scheduler
from brian2.core.variables import Variables
from brian2.units.allunits import second, hertz
from brian2.units.fundamentalunits import Unit, Quantity
from brian2.groups.group import CodeRunner, Group

__all__ = ['PopulationRateMonitor']


class PopulationRateMonitor(Group, CodeRunner):
    '''
    Record instantaneous firing rates, averaged across neurons from a
    `NeuronGroup` or other spike source.

    Parameters
    ----------
    source : (`NeuronGroup`, `SpikeSource`)
        The source of spikes to record.
    name : str, optional
        A unique name for the object, otherwise will use
        ``source.name+'_ratemonitor_0'``, etc.
    codeobj_class : class, optional
        The `CodeObject` class to run code with.
    '''
    def __init__(self, source, name='ratemonitor*',
                 codeobj_class=None):

        #: The group we are recording from
        self.source = source

        scheduler = Scheduler(clock=source.clock, when='end')

        self.codeobj_class = codeobj_class
        CodeRunner.__init__(self, group=self, template='ratemonitor',
                            when=scheduler, name=name)

        self.variables = Variables(self)
        self.variables.add_reference('_spikespace',
                                     source.variables['_spikespace'])
        self.variables.add_reference('_clock_t', source.variables['t'])
        self.variables.add_reference('_clock_dt', source.variables['dt'])
        self.variables.add_dynamic_array('rate', size=0, unit=hertz,
                                         constant_size=False)
        self.variables.add_dynamic_array('t', size=0, unit=second,
                                         constant_size=False)
        self.variables.add_reference('_num_source_neurons',
                                     source.variables['N'])
        self.variables.add_attribute_variable('N', unit=Unit(1), obj=self,
                                              attribute='_N', dtype=np.int32)
        self._N = 0

        self._enable_group_attributes()

    def resize(self, new_size):
        self.variables['rate'].resize(new_size)
        self.variables['t'].resize(new_size)
        self._N = new_size

    def __len__(self):
        return self._N

    def reinit(self):
        '''
        Clears all recorded rates
        '''
        raise NotImplementedError()

    def __repr__(self):
        description = '<{classname}, recording {source}>'
        return description.format(classname=self.__class__.__name__,
                                  source=self.source.name)
