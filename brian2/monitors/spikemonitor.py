import numpy as np

from brian2.core.scheduler import Scheduler
from brian2.core.variables import Variables
from brian2.units.allunits import second
from brian2.units.fundamentalunits import Unit, Quantity
from brian2.groups.group import CodeRunner, Group

__all__ = ['SpikeMonitor']


class SpikeMonitor(Group, CodeRunner):
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
        #: The source we are recording from
        self.source = source

        # run by default on source clock at the end
        scheduler = Scheduler(when)
        if not scheduler.defined_clock:
            scheduler.clock = source.clock
        if not scheduler.defined_when:
            scheduler.when = 'end'

        self.codeobj_class = codeobj_class
        CodeRunner.__init__(self, group=self, template='spikemonitor',
                            name=name, when=scheduler)

        # Handle subgroups correctly
        start = getattr(source, 'start', 0)
        stop = getattr(source, 'stop', len(source))

        self.variables = Variables(self)
        self.variables.add_clock_variables(scheduler.clock, prefix='_clock_')
        self.variables.add_reference('_spikespace', source.variables['_spikespace'])
        self.variables.add_dynamic_array('i', size=0, unit=Unit(1),
                                         dtype=np.int32, constant_size=False)
        self.variables.add_dynamic_array('t', size=0, unit=second,
                                         constant_size=False)
        self.variables.add_array('_count', size=len(source), unit=Unit(1),
                                 dtype=np.int32)
        self.variables.add_constant('_source_start', Unit(1), start)
        self.variables.add_constant('_source_stop', Unit(1), stop)
        self.variables.add_attribute_variable('N', unit=Unit(1), obj=self,
                                              attribute='_N', dtype=np.int32)
        self._N = 0

        self._enable_group_attributes()

    def resize(self, new_size):
        self.variables['i'].resize(new_size)
        self.variables['t'].resize(new_size)
        self._N = new_size

    def __len__(self):
        return self._N

    def reinit(self):
        '''
        Clears all recorded spikes
        '''
        raise NotImplementedError()

    # TODO: Maybe there's a more elegant solution for the count attribute?
    @property
    def count(self):
        return self.variables['_count'].get_value().copy()

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
        Returns the total number of recorded spikes
        '''
        return self._N

    def __repr__(self):
        description = '<{classname}, recording {source}>'
        return description.format(classname=self.__class__.__name__,
                                  source=self.group.name)
