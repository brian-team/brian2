'''
Module defining `SpikeGeneratorGroup`.
'''
import numpy as np

from brian2.core.spikesource import SpikeSource
from brian2.units.fundamentalunits import check_units, Unit
from brian2.units.allunits import second
from brian2.core.variables import Variables
from brian2.groups.group import CodeRunner

from .group import Group

__all__ = ['SpikeGeneratorGroup']


class SpikeGeneratorGroup(Group, CodeRunner, SpikeSource):
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
    dt : `Quantity`, optional
        The time step to be used for the simulation. Cannot be combined with
        the `clock` argument.
    clock : `Clock`, optional
        The update clock to be used. If neither a clock, nor the `dt` argument
        is specified, the `defaultclock` will be used.
    when : str, optional
        When to run within a time step, defaults to the ``'thresholds'`` slot.
    order : int, optional
        The priority of of this group for operations occurring at the same time
        step and in the same scheduling slot. Defaults to 0.
    sorted : bool, optional
        Whether the given indices and times are already sorted. Set to ``True``
        if your events are already sorted (first by spike time, then by index),
        this can save significant time at construction if your arrays contain
        large numbers of spikes. Defaults to ``False``.

    Notes
    -----
    * In a time step, `SpikeGeneratorGroup` emits all spikes that happened
      at :math:`t-dt < t_{spike} \leq t`. This might lead to unexpected
      or missing spikes if you change the time step dt between runs.
    * `SpikeGeneratorGroup` does not currently raise any warning if a neuron
      spikes more that once during a time step, but other code (e.g. for
      synaptic propagation) might assume that neurons only spike once per
      time step and will therefore not work properly.
    * If `sorted` is set to ``True``, the given arrays will not be copied.
    '''

    @check_units(N=1, indices=1, times=second)
    def __init__(self, N, indices, times, dt=None, clock=None,
                 when='thresholds', order=0, sorted=False,
                 name='spikegeneratorgroup*', codeobj_class=None):

        Group.__init__(self, dt=dt, clock=clock, when=when, order=order, name=name)

        self.codeobj_class = codeobj_class

        if N < 1 or int(N) != N:
            raise ValueError('N has to be an integer >=1.')

        if len(indices) != len(times):
            raise ValueError(('Length of the indices and times array must '
                              'match, but %d != %d') % (len(indices),
                                                        len(times)))

        self.start = 0
        self.stop = N

        if not sorted:
            # sort times and indices first by time, then by indices
            rec = np.rec.fromarrays([times, indices], names=['t', 'i'])
            rec.sort()
            times = np.ascontiguousarray(rec.t)
            indices = np.ascontiguousarray(rec.i)

        self.variables = Variables(self)

        # standard variables
        self.variables.add_constant('N', unit=Unit(1), value=N)
        self.variables.add_arange('i', N)
        self.variables.add_arange('spike_number', len(indices))
        self.variables.add_array('neuron_index', values=indices,
                                 size=len(indices), unit=Unit(1),
                                 dtype=np.int32, index='spike_number',
                                 read_only=True)
        self.variables.add_array('spike_time', values=times, size=len(times),
                                 unit=second, index='spike_number',
                                 read_only=True)
        self.variables.add_array('_spikespace', size=N+1, unit=Unit(1),
                                 dtype=np.int32)
        self.variables.create_clock_variables(self._clock)

        # Activate name attribute access
        self._enable_group_attributes()

        CodeRunner.__init__(self, self,
                            code='',
                            template='spikegenerator',
                            clock=self._clock,
                            when=when,
                            order=order,
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
