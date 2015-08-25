import numpy as np

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

    Notes
    -----
    Currently, this monitor can only monitor the instantaneous firing rates at
    each time step of the source clock. Any binning/smoothing of the firing
    rates has to be done manually afterwards.
    '''
    invalidates_magic_network = False
    add_to_magic_network = True
    def __init__(self, source, name='ratemonitor*',
                 codeobj_class=None):

        #: The group we are recording from
        self.source = source

        self.codeobj_class = codeobj_class
        CodeRunner.__init__(self, group=self, code='', template='ratemonitor',
                            clock=source.clock, when='end', order=0, name=name)

        self.add_dependency(source)

        self.variables = Variables(self)
        # Handle subgroups correctly
        start = getattr(source, 'start', 0)
        stop = getattr(source, 'stop', len(source))
        self.variables.add_constant('_source_start', Unit(1), start)
        self.variables.add_constant('_source_stop', Unit(1), stop)
        self.variables.add_reference('_spikespace', source)
        self.variables.add_dynamic_array('rate', size=0, unit=hertz,
                                         constant_size=False)
        self.variables.add_dynamic_array('t', size=0, unit=second,
                                         constant_size=False)
        self.variables.add_reference('_num_source_neurons', source, 'N')
        self.variables.add_array('N', unit=Unit(1), dtype=np.int32, size=1,
                                 scalar=True, read_only=True)
        self.variables.create_clock_variables(self._clock,
                                              prefix='_clock_')
        self._enable_group_attributes()

    def resize(self, new_size):
        self.variables['N'].set_value(new_size)
        self.variables['rate'].resize(new_size)
        self.variables['t'].resize(new_size)

    def reinit(self):
        '''
        Clears all recorded rates
        '''
        raise NotImplementedError()

    def __repr__(self):
        description = '<{classname}, recording {source}>'
        return description.format(classname=self.__class__.__name__,
                                  source=self.source.name)
