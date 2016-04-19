import numbers

import numpy as np

from brian2.core.variables import Variables
from brian2.core.names import Nameable
from brian2.core.spikesource import SpikeSource
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
    source : `NeuronGroup`, `SpikeSource`
        The source of events to record.
    event : str
        The name of the event to record
    variables : str or sequence of str, optional
        Which variables to record at the time of the event (in addition to the
        index of the neuron). Can be the name of a variable or a list of names.
    record : bool, optional
        Whether or not to record each event in `i` and `t` (the `count` will
        always be recorded). Defaults to ``True``.
    when : str, optional
        When to record the events, by default records events in the same slot
        where the event is emitted.
    order : int, optional
        The priority of of this group for operations occurring at the same time
        step and in the same scheduling slot. Defaults to the order where the
        event is emitted + 1, i.e. it will be recorded directly afterwards.
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

    def __init__(self, source, event, variables=None, record=True,
                 when=None, order=None, name='eventmonitor*',
                 codeobj_class=None):
        if not isinstance(source, SpikeSource):
            raise TypeError(('%s can only monitor groups producing spikes '
                             '(such as NeuronGroup), but the given argument '
                             'is of type %s.') % (self.__class__.__name__,
                                                  type(source)))
        #: The source we are recording from
        self.source = source
        #: Whether to record times and indices of events
        self.record = record

        if when is None:
            if order is not None:
                raise ValueError('Cannot specify order if when is not specified.')
            if hasattr(source, 'thresholder'):
                parent_obj = source.thresholder[event]
            else:
                parent_obj = source
            when = parent_obj.when
            order = parent_obj.order + 1
        elif order is None:
            order = 0

        #: The event that we are listening to
        self.event = event

        if variables is None:
            variables = {}
        elif isinstance(variables, basestring):
            variables = {variables}

        #: The additional variables that will be recorded
        self.record_variables = set(variables)

        for variable in variables:
            if variable not in source.variables:
                raise ValueError(("'%s' is not a variable of the recorded "
                                  "group" % variable))

        if self.record:
            self.record_variables |= {'i', 't'}

        # Some dummy code so that code generation takes care of the indexing
        # and subexpressions
        code = ['_to_record_%s = _source_%s' % (v, v)
                for v in self.record_variables]
        code = '\n'.join(code)

        self.codeobj_class = codeobj_class

        # Since this now works for general events not only spikes, we have to
        # pass the information about which variable to use to the template,
        # it can not longer simply refer to "_spikespace"
        eventspace_name = '_{}space'.format(event)

        # Handle subgroups correctly
        start = getattr(source, 'start', 0)
        stop = getattr(source, 'stop', len(source))

        Nameable.__init__(self, name=name)

        self.variables = Variables(self)
        self.variables.add_reference(eventspace_name, source)

        for variable in self.record_variables:
            source_var = source.variables[variable]
            self.variables.add_reference('_source_%s' % variable,
                                         source, variable)
            self.variables.add_auxiliary_variable('_to_record_%s' % variable,
                                                   unit=source_var.unit,
                                                   dtype=source_var.dtype)
            self.variables.add_dynamic_array(variable, size=0,
                                             unit=source_var.unit,
                                             dtype=source_var.dtype,
                                             read_only=True)
        self.variables.add_arange('_source_idx', size=len(source))
        self.variables.add_array('count', size=len(source), unit=Unit(1),
                                 dtype=np.int32, read_only=True,
                                 index='_source_idx')
        self.variables.add_constant('_source_start', Unit(1), start)
        self.variables.add_constant('_source_stop', Unit(1), stop)
        self.variables.add_array('N', unit=Unit(1), size=1, dtype=np.int32,
                                 read_only=True, scalar=True)

        record_variables = {varname: self.variables[varname]
                            for varname in self.record_variables}
        template_kwds = {'eventspace_variable': source.variables[eventspace_name],
                         'record_variables': record_variables,
                         'record': self.record}
        needed_variables = {eventspace_name} | self.record_variables
        CodeRunner.__init__(self, group=self, code=code, template='spikemonitor',
                            name=None,  # The name has already been initialized
                            clock=source.clock, when=when,
                            order=order, needed_variables=needed_variables,
                            template_kwds=template_kwds)

        self.variables.create_clock_variables(self._clock,
                                              prefix='_clock_')

        self.add_dependency(source)
        self._enable_group_attributes()

    def resize(self, new_size):
        # Note that this does not set N, this has to be done in the template
        # since we use a restricted pointer to access it (which promises that
        # we only change the value through this pointer)
        for variable in self.record_variables:
            self.variables[variable].resize(new_size)

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
        if not self.record:
            raise AttributeError('Indices and times have not been recorded.'
                                 'Set the record argument to True to record '
                                 'them.')
        return self.i, self.t

    @property
    def it_(self):
        '''
        Returns the pair (`i`, `t_`).
        '''
        if not self.record:
            raise AttributeError('Indices and times have not been recorded.'
                                 'Set the record argument to True to record '
                                 'them.')

        return self.i, self.t_

    def _values_dict(self, first_pos, sort_indices, used_indices, var):
        sorted_values = self.state(var, use_units=False)[sort_indices]
        dim = self.variables[var].unit.dim
        event_values = {}
        current_pos = 0  # position in the all_indices array
        for idx in xrange(len(self.source)):
            if current_pos < len(used_indices) and used_indices[current_pos] == idx:
                if current_pos < len(used_indices) - 1:
                    event_values[idx] = Quantity(sorted_values[
                                                 first_pos[current_pos]:
                                                 first_pos[current_pos + 1]],
                                                 dim=dim, copy=False)
                else:
                    event_values[idx] = Quantity(
                        sorted_values[first_pos[current_pos]:],
                        dim=dim, copy=False)
                current_pos += 1
            else:
                event_values[idx] = Quantity([], dim=dim)
        return event_values

    def values(self, var):
        '''
        Return a dictionary mapping neuron indices to arrays of variable values
        at the time of the events (sorted by time).
        Parameters
        ----------
        var : str
            The name of the variable.

        Returns
        -------
        values : dict
            Dictionary mapping each neuron index to an array of variable
            values at the time of the events

        Examples
        --------
        >>> from brian2 import *
        >>> G = NeuronGroup(2, """dv/dt = 100*Hz : 1
        ...                       v_th : 1""", threshold='v>v_th', reset='v=0')
        >>> G.v_th = [0.5, 1]
        >>> mon = EventMonitor(G, event='spike', variables='v')
        >>> run(20*ms)
        >>> v_values = mon.values('v')
        >>> v_values[0]
        array([ 0.5,  0.5,  0.5,  0.5])
        >>> v_values[1]
        array([ 1.,  1.])
        '''
        if not self.record:
            raise AttributeError('Indices and times have not been recorded.'
                                 'Set the record argument to True to record '
                                 'them.')
        indices = self.i[:]
        sort_indices = np.argsort(indices)
        used_indices, first_pos = np.unique(self.i[:][sort_indices],
                                            return_index=True)
        return self._values_dict(first_pos, sort_indices, used_indices, var)

    def all_values(self):
        '''
        Return a dictionary mapping recorded variable names (including ``t``)
        to a dictionary mapping neuron indices to arrays of variable values at
        the time of the events (sorted by time). This is equivalent to (but more
        efficient than) calling `values` for each variable and storing the
        result in a dictionary.

        Returns
        -------
        all_values : dict
            Dictionary mapping variable names to dictionaries which themselves
            are mapping neuron indicies to arrays of variable values at the
            time of the events.

        Examples
        --------
        >>> from brian2 import *
        >>> G = NeuronGroup(2, """dv/dt = 100*Hz : 1
        ...                       v_th : 1""", threshold='v>v_th', reset='v=0')
        >>> G.v_th = [0.5, 1]
        >>> mon = EventMonitor(G, event='spike', variables='v')
        >>> run(20*ms)
        >>> all_values = mon.all_values()
        >>> all_values['t'][0]
        array([  4.9,   9.9,  14.9,  19.9]) * msecond
        >>> all_values['v'][0]
        array([ 0.5,  0.5,  0.5,  0.5])
        '''
        if not self.record:
            raise AttributeError('Indices and times have not been recorded.'
                                 'Set the record argument to True to record '
                                 'them.')
        indices = self.i[:]
        sort_indices = np.argsort(indices)
        used_indices, first_pos = np.unique(self.i[:][sort_indices],
                                            return_index=True)
        all_values_dict = {}
        for varname in self.record_variables - {'i'}:
            all_values_dict[varname] = self._values_dict(first_pos,
                                                         sort_indices,
                                                         used_indices,
                                                         varname)
        return all_values_dict

    def event_trains(self):
        '''
        Return a dictionary mapping event indices to arrays of event times.
        Equivalent to calling ``values('t')``.

        Returns
        -------
        event_trains : dict
            Dictionary that stores an array with the event times for each
            neuron index.

        See Also
        --------
        SpikeMonitor.spike_trains
        '''
        return self.values('t')

    @property
    def num_events(self):
        '''
        Returns the total number of recorded events.
        '''
        return self.N[:]

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
    method. If you record additional variables with the ``variables`` argument,
    these variables can be accessed by their name (see Examples).

    Parameters
    ----------
    source : (`NeuronGroup`, `SpikeSource`)
        The source of spikes to record.
    variables : str or sequence of str, optional
        Which variables to record at the time of the spike (in addition to the
        index of the neuron). Can be the name of a variable or a list of names.
    record : bool, optional
        Whether or not to record each spike in `i` and `t` (the `count` will
        always be recorded). Defaults to ``True``.
    when : str, optional
        When to record the events, by default records events in the same slot
        where the event is emitted.
    order : int, optional
        The priority of of this group for operations occurring at the same time
        step and in the same scheduling slot. Defaults to the order where the
        event is emitted + 1, i.e. it will be recorded directly afterwards.
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
    >>> net = Network(spikes, spike_mon)
    >>> net.run(3*ms)
    >>> print(spike_mon.i[:])
    [0 1 2]
    >>> print(spike_mon.t[:])
    [ 0.  1.  2.] ms
    >>> print(spike_mon.t_[:])
    [ 0.     0.001  0.002]
    >>> G = NeuronGroup(1, """dv/dt = (1 - v)/(10*ms) : 1
    ...                       dv_th/dt = (0.5 - v_th)/(20*ms) : 1""",
    ...                 threshold='v>v_th',
    ...                 reset='v = 0; v_th += 0.1')
    >>> crossings = SpikeMonitor(G, variables='v', name='crossings')
    >>> net = Network(G, crossings)
    >>> net.run(10*ms)
    >>> crossings.t
    <crossings.t: array([ 0. ,  1.4,  4.6,  9.7]) * msecond>
    >>> crossings.v
    <crossings.v: array([ 0.00995017,  0.13064176,  0.27385096,  0.39950442])>
    '''
    def __init__(self, source, variables=None, record=True, when=None,
                 order=None, name='spikemonitor*', codeobj_class=None):
        super(SpikeMonitor, self).__init__(source, event='spike',
                                           variables=variables, record=record,
                                           when=when, order=order, name=name,
                                           codeobj_class=codeobj_class)

    @property
    def num_spikes(self):
        '''
        Returns the total number of recorded spikes.
        '''
        return self.num_events

    # We "re-implement" the following functions only to get more specific
    # doc strings (and to make sure that the methods are included in the
    # reference documentation for SpikeMonitor).

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

    def values(self, var):
        '''
        Return a dictionary mapping neuron indices to arrays of variable values
        at the time of the spikes (sorted by time).

        Parameters
        ----------
        var : str
            The name of the variable.

        Returns
        -------
        values : dict
            Dictionary mapping each neuron index to an array of variable
            values at the time of the spikes.

        Examples
        --------
        >>> from brian2 import *
        >>> G = NeuronGroup(2, """dv/dt = 100*Hz : 1
        ...                       v_th : 1""", threshold='v>v_th', reset='v=0')
        >>> G.v_th = [0.5, 1]
        >>> mon = SpikeMonitor(G, variables='v')
        >>> run(20*ms)
        >>> v_values = mon.values('v')
        >>> v_values[0]
        array([ 0.5,  0.5,  0.5,  0.5])
        >>> v_values[1]
        array([ 1.,  1.])
        '''
        return super(SpikeMonitor, self).values(var)

    def all_values(self):
        '''
        Return a dictionary mapping recorded variable names (including ``t``)
        to a dictionary mapping neuron indices to arrays of variable values at
        the time of the spikes (sorted by time). This is equivalent to (but more
        efficient than) calling `values` for each variable and storing the
        result in a dictionary.

        Returns
        -------
        all_values : dict
            Dictionary mapping variable names to dictionaries which themselves
            are mapping neuron indicies to arrays of variable values at the
            time of the spikes.

        Examples
        --------
        >>> from brian2 import *
        >>> G = NeuronGroup(2, """dv/dt = 100*Hz : 1
        ...                       v_th : 1""", threshold='v>v_th', reset='v=0')
        >>> G.v_th = [0.5, 1]
        >>> mon = SpikeMonitor(G, variables='v')
        >>> run(20*ms)
        >>> all_values = mon.all_values()
        >>> all_values['t'][0]
        array([  4.9,   9.9,  14.9,  19.9]) * msecond
        >>> all_values['v'][0]
        array([ 0.5,  0.5,  0.5,  0.5])
        '''
        return super(SpikeMonitor, self).all_values()

    def __repr__(self):
        description = '<{classname}, recording from {source}>'
        return description.format(classname=self.__class__.__name__,
                                  source=self.group.name)
