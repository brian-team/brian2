import collections
import itertools
import numbers

import numpy as np

from brian2.core.variables import Variables
from brian2.units.allunits import second
from brian2.units.fundamentalunits import Unit, Quantity
from brian2.groups.group import CodeRunner, Group

__all__ = ['SpikeMonitor']


class SpikeMonitor(Group, CodeRunner, collections.Mapping, collections.Hashable):
    '''
    Record spikes from a `NeuronGroup` or other spike source.

    The recorded spikes can be accessed in various ways (see Examples below):
    the attributes `~SpikeMonitor.i` and `~SpikeMonitor.t` store all the indices
    and spike times, respectively. The `SpikeMonitor` object can also be
    accessed like a dictionary mapping neuron indices to arrays of spike times.
    Note that if the `SpikeMonitor` stores a large number of spikes, getting the
    spike times using the dictionary indexing can be slow.

    Parameters
    ----------
    source : (`NeuronGroup`, `SpikeSource`)
        The source of spikes to record.
    record : bool
        Whether or not to record each spike in `i` and `t` (the `count` will
        always be recorded).
    when : str, optional
        When to record the spikes, by default records spikes in the slot
        ``'end'``.
    order : int, optional
        The priority of of this group for operations occurring at the same time
        step and in the same scheduling slot. Defaults to 0.
    name : str, optional
        A unique name for the object, otherwise will use
        ``source.name+'_spikemonitor_0'``, etc.
    codeobj_class : class, optional
        The `CodeObject` class to run code with.

    Examples
    --------
    >>> from brian2 import *
    >>> spikes = SpikeGeneratorGroup(3, [0, 1, 2], [0, 1, 2]*ms)
    >>> spike_mon = SpikeMonitor(spikes)
    >>> run(2*ms)
    >>> print(spike_mon.i[:])
    [0 1]
    >>> print(spike_mon.t[:])
    [ 0.  1.] ms
    >>> print(spike_mon.t_[:])
    [ 0.     0.001]
    >>> print(spike_mon[0])
    [ 0.] s
    >>> print(spike_mon.items())
    [(0, array([ 0.]) * second), (1, array([ 1.]) * msecond), (2, array([], dtype=float64) * second)]
    '''
    invalidates_magic_network = False
    add_to_magic_network = True

    def __init__(self, source, record=True, when='end', order=0,
                 name='spikemonitor*', codeobj_class=None):
        self.record = bool(record)
        #: The source we are recording from
        self.source =source

        self.codeobj_class = codeobj_class
        CodeRunner.__init__(self, group=self, code='', template='spikemonitor',
                            name=name, clock=source.clock, when=when,
                            order=order)

        self.add_dependency(source)

        # Handle subgroups correctly
        start = getattr(source, 'start', 0)
        stop = getattr(source, 'stop', len(source))

        self.variables = Variables(self)
        self.variables.add_reference('_spikespace', source)
        self.variables.add_dynamic_array('i', size=0, unit=Unit(1),
                                         dtype=np.int32, constant_size=False)
        self.variables.add_dynamic_array('t', size=0, unit=second,
                                         constant_size=False)
        self.variables.add_arange('_source_i', size=len(source))
        self.variables.add_array('_count', size=len(source), unit=Unit(1),
                                 dtype=np.int32, read_only=True,
                                 index='_source_i')
        self.variables.add_constant('_source_start', Unit(1), start)
        self.variables.add_constant('_source_stop', Unit(1), stop)
        self.variables.add_attribute_variable('N', unit=Unit(1), obj=self,
                                              attribute='_N', dtype=np.int32)
        self.variables.create_clock_variables(self._clock,
                                              prefix='_clock_')
        self._enable_group_attributes()

    @property
    def _N(self):
        return len(self.variables['t'].get_value())

    def resize(self, new_size):
        self.variables['i'].resize(new_size)
        self.variables['t'].resize(new_size)

    def __len__(self):
        return self._N

    def __getitem__(self, item):
        if not isinstance(item, numbers.Integral):
            raise TypeError(('Index has to be an integer, is type %s '
                             'instead.') % type(item))
        if item < 0 or item >= len(self.source):
            raise IndexError(('Index has to be between 0 and %d, was '
                              '%d.') % (len(self.source), item))
        return Quantity(self.t_[:][np.where(self.i[:] == item)],
                        dim=second.dim, copy=False)

    def __iter__(self):
        return iter(xrange(len(self.source)))

    def __hash__(self):
        return id(self)

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
