import numbers

import numpy as np

from brian2.core.variables import Variables
from brian2.units.allunits import second
from brian2.units.fundamentalunits import Unit, Quantity
from brian2.groups.group import CodeRunner, Group

__all__ = ['EventMonitor', 'SpikeMonitor']


class EventMonitor(Group, CodeRunner):
    '''
    Record events from a `NeuronGroup` or another event source.

    The recorded events can be accessed in various ways:
    the attributes `~EventMonitor.i` and `~EventMonitor.t` store all the indices
    and event times, respectively. Alternatively, you can get a dictionary
    mapping neuron indices to event trains, by calling the `event_trains`
    method.

    Parameters
    ----------
    source : `NeuronGroup`
        The source of events to record.
    record : bool
        Whether or not to record each event in `i` and `t` (the `count` will
        always be recorded).
    when : str, optional
        When to record the events, by default records events in the slot
        ``'end'``.
    order : int, optional
        The priority of of this group for operations occurring at the same time
        step and in the same scheduling slot. Defaults to 0.
    name : str, optional
        A unique name for the object, otherwise will use
        ``source.name+'_eventmonitor_0'``, etc.
    codeobj_class : class, optional
        The `CodeObject` class to run code with.

    See Also
    --------
    SpikeMonitor
    '''
    invalidates_magic_network = False
    add_to_magic_network = True

    def __init__(self, source, event, when='end', order=0,
                 name='eventmonitor*', codeobj_class=None):
        #: The source we are recording from
        self.source = source

        #: The event that we are listening to
        self.event = event

        self.codeobj_class = codeobj_class
        # Since this now works for general events not only spikes, we have to
        # pass the information about which variable to use to the template,
        # it can not longer simply refer to "_spikespace"
        eventspace_name = '_{}space'.format(event)
        template_kwds = {'eventspace_variable': source.variables[eventspace_name]}
        needed_variables= [eventspace_name]
        CodeRunner.__init__(self, group=self, code='', template='spikemonitor',
                            name=name, clock=source.clock, when=when,
                            order=order, needed_variables=needed_variables,
                            template_kwds=template_kwds)

        self.add_dependency(source)

        # Handle subgroups correctly
        start = getattr(source, 'start', 0)
        stop = getattr(source, 'stop', len(source))

        self.variables = Variables(self)
        self.variables.add_reference(eventspace_name, source)
        self.variables.add_dynamic_array('i', size=0, unit=Unit(1),
                                         dtype=np.int32, constant_size=False)
        self.variables.add_dynamic_array('t', size=0, unit=second,
                                         constant_size=False)
        self.variables.add_arange('_source_i', size=len(source))
        self.variables.add_array('count', size=len(source), unit=Unit(1),
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

    def reinit(self):
        '''
        Clears all recorded spikes
        '''
        raise NotImplementedError()

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

    def event_trains(self):
        '''
        Return a dictionary mapping event indices to arrays of event times.

        Returns
        -------
        event_trains : dict
            Dictionary that stores an array with the event times for each
            neuron index.

        See Also
        --------
        SpikeMonitor.spike_trains
        '''
        indices = self.i[:]
        sort_indices = np.argsort(indices)
        used_indices, first_pos = np.unique(self.i[:][sort_indices],
                                            return_index=True)
        sorted_times = self.t_[:][sort_indices]

        event_times = {}
        current_pos = 0  # position in the all_indices array
        for idx in xrange(len(self.source)):
            if current_pos < len(used_indices) and used_indices[current_pos] == idx:
                if current_pos < len(used_indices)-1:
                    event_times[idx] = Quantity(sorted_times[first_pos[current_pos]:first_pos[current_pos+1]],
                                                dim=second.dim, copy=False)
                else:
                    event_times[idx] = Quantity(sorted_times[first_pos[current_pos]:],
                                                dim=second.dim, copy=False)
                current_pos += 1
            else:
                event_times[idx] = Quantity([], dim=second.dim)

        return event_times

    @property
    def num_events(self):
        '''
        Returns the total number of recorded events.
        '''
        return self._N

    def __repr__(self):
        description = '<{classname}, recording event "{event}" from {source}>'
        return description.format(classname=self.__class__.__name__,
                                  event=self.event,
                                  source=self.group.name)

class SpikeMonitor(EventMonitor):
    '''
    Record spikes from a `NeuronGroup` or other spike source.

    The recorded spikes can be accessed in various ways (see Examples below):
    the attributes `~SpikeMonitor.i` and `~SpikeMonitor.t` store all the indices
    and spike times, respectively. Alternatively, you can get a dictionary
    mapping neuron indices to spike trains, by calling the `spike_trains`
    method.

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
    >>> run(3*ms)
    >>> print(spike_mon.i[:])
    [0 1 2]
    >>> print(spike_mon.t[:])
    [ 0.  1.  2.] ms
    >>> print(spike_mon.t_[:])
    [ 0.     0.001  0.002]
    '''
    def __init__(self, source, when='end', order=0,
             name='spikemonitor*', codeobj_class=None):
        super(SpikeMonitor, self).__init__(source, event='spike', when=when,
                                           order=order, name=name,
                                           codeobj_class=codeobj_class)

    @property
    def num_spikes(self):
        '''
        Returns the total number of recorded spikes.
        '''
        return self.num_events

    def spike_trains(self):
        '''
        Return a dictionary mapping spike indices to arrays of spike times.

        Returns
        -------
        spike_trains : dict
            Dictionary that stores an array with the spike times for each
            neuron index.

        Examples
        --------
        >>> from brian2 import *
        >>> spikes = SpikeGeneratorGroup(3, [0, 1, 2], [0, 1, 2]*ms)
        >>> spike_mon = SpikeMonitor(spikes)
        >>> run(3*ms)
        >>> spike_trains = spike_mon.spike_trains()
        >>> spike_trains[1]
        array([ 1.]) * msecond
        '''
        return self.event_trains()

    def __repr__(self):
        description = '<{classname}, recording from {source}>'
        return description.format(classname=self.__class__.__name__,
                                  source=self.group.name)
