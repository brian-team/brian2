'''
Module defining `SpikeGeneratorGroup`.
'''
import numpy as np

from brian2.core.spikesource import SpikeSource
from brian2.core.scheduler import Scheduler
from brian2.units.fundamentalunits import check_units, Unit
from brian2.units.allunits import second
from brian2.core.variables import Variables
from brian2.groups.group import CodeRunner

from .group import Group

__all__ = ['SpikeGeneratorGroup']


class SpikeGeneratorGroup(Group, CodeRunner, SpikeSource):
    @check_units(N=1, indices=1, times=second)
    def __init__(self, N, indices, times, when=None,
                 name='spikegeneratorgroup*', codeobj_class=None):
        '''
        A group emitting spikes at given times.

        Parameters
        ----------
        N : int
            The number of "neurons" in this group
        indices : array of integers
            The indices of the spiking cells
        times : `Quantity`
            The spike times for the cells given in `indices`. Has to have the
            same length as `indices`.
        when : `Scheduler`
            When to update this group
        '''
        if when is None:
            when = Scheduler(when='thresholds')
        Group.__init__(self, when=when, name=name)

        self.codeobj_class = codeobj_class

        if N < 1 or int(N) != N:
            raise ValueError('N has to be an integer >=1.')

        if len(indices) != len(times):
            raise ValueError(('Length of the indices and times array must '
                              'match, but %d != %d') % (len(indices),
                                                        len(times)))

        self.start = 0
        self.stop = N

        sort_indices = np.argsort(times)
        times = times[sort_indices]
        indices = indices[sort_indices]

        self.variables = Variables(self)

        # standard variables
        self.variables.add_clock_variables(self.clock)
        self.variables.add_constant('N', unit=Unit(1), value=N)
        self.variables.add_arange('i', N)
        self.variables.add_arange('spike_number', len(indices))
        self.variables.add_array('neuron_index', size=len(indices),
                                 unit=Unit(1), dtype=np.int32,
                                 index='spike_number')
        self.variables.add_array('spike_time', size=len(times), unit=second,
                                 index='spike_number')
        self.variables.add_array('_spikespace', size=N+1, unit=Unit(1),
                                 dtype=np.int32)

        # Activate name attribute access
        self._enable_group_attributes()

        # Set the arrays
        self.neuron_index.set_item(slice(None), indices)
        self.spike_time.set_item(slice(None), times)

        CodeRunner.__init__(self, self,
                            'spikegenerator',
                            when=when,
                            name=None)

    @property
    def spikes(self):
        '''
        The spikes returned by the most recent thresholding operation.
        '''
        # Note that we have to directly access the ArrayVariable object here
        # instead of using the Group mechanism by accessing self._spikespace
        # Using the latter would cut _spikespace to the length of the group
        spikespace = self.variables['_spikespace'].get_value()
        return spikespace[:spikespace[-1]]

    def __len__(self):
        return self.N

    def __repr__(self):
        return ('{cls}({N}, indices=<length {l} array>, '
                'times=<length {l} array>').format(cls=self.__class__.__name__,
                                                   N=self.N,
                                                   l=self.variables['neuron_index'].size)
