'''
Module defining the `Network` object, the basis of all simulation runs.

Preferences
-----------

.. document_brian_prefs:: core.network

'''
import os
import sys
import time
from collections import defaultdict, Sequence, Counter, Mapping, namedtuple
import cPickle as pickle

from brian2.synapses.synapses import SummedVariableUpdater
from brian2.utils.logger import get_logger
from brian2.core.names import Nameable
from brian2.core.base import BrianObject, brian_object_exception
from brian2.core.clocks import Clock, defaultclock
from brian2.devices.device import get_device, all_devices
from brian2.groups.group import Group
from brian2.units.fundamentalunits import check_units, Quantity
from brian2.units.allunits import second, msecond
from brian2.core.preferences import prefs, BrianPreference
from brian2.core.namespace import get_local_namespace
from .base import device_override

__all__ = ['Network', 'profiling_summary', 'scheduling_summary']


logger = get_logger(__name__)


prefs.register_preferences('core.network', 'Network preferences',
                           default_schedule=BrianPreference(
                               default=['start',
                                        'groups',
                                        'thresholds',
                                        'synapses',
                                        'resets',
                                        'end',
                                        ],
                               docs='''
                               Default schedule used for networks that
                               don't specify a schedule.
                               '''
                           )
                           )

def _format_time(time_in_s):
    '''
    Helper function to format time in seconds, minutes, hours, days, depending
    on the magnitude.

    Examples
    --------
    >>> from brian2.core.network import _format_time
    >>> _format_time(12345)
    '3h 25m 45s'
    >>> _format_time(123)
    '2m 3s'
    >>> _format_time(12.5)
    '12s'
    >>> _format_time(.5)
    '< 1s'

    '''
    divisors = [24*60*60, 60*60, 60, 1]
    letters = ['d', 'h', 'm', 's']
    remaining = time_in_s
    text = ''
    for divisor, letter in zip(divisors, letters):
        time_to_represent = int(remaining / divisor)
        remaining -= time_to_represent * divisor
        if time_to_represent > 0 or len(text):
            if len(text):
                text += ' '
            text += '%d%s' % (time_to_represent, letter)

    # less than one second
    if len(text) == 0:
        text = '< 1s'

    return text


class TextReport(object):
    '''
    Helper object to report simulation progress in `Network.run`.

    Parameters
    ----------
    stream : file
        The stream to write to, commonly `sys.stdout` or `sys.stderr`.
    '''
    def __init__(self, stream):
        self.stream = stream

    def __call__(self, elapsed, completed, start, duration):
        if completed == 0.0:
            self.stream.write(('Starting simulation at t=%s for a duration of '
                               '%s\n') % (start, duration))
        else:
            report_msg = ('{t} ({percent}%) simulated in '
                          '{real_t}').format(t=completed*duration,
                                             percent=int(completed*100.),
                                             real_t=_format_time(float(elapsed)))
            if completed < 1.0:
                remaining = int(round((1-completed)/completed*float(elapsed)))
                remaining_msg = (', estimated {remaining} '
                                 'remaining.\n').format(remaining=_format_time(remaining))
            else:
                remaining_msg = '\n'

            self.stream.write(report_msg + remaining_msg)

        # Flush the stream, this is useful if stream is a file
        self.stream.flush()


def _format_table(header, values, cell_formats):
    # table = [header] + values
    table_format = len(values)*[cell_formats]
    col_widths = [max(len(format.format(cell, 0))
                      for format, cell in zip(col_format, col))
                  for col_format, col in zip(zip(*([len(header)*['{}']] + table_format)),
                                             zip(*([header] + values)))]
    line = '-+-'.join('-'*width for width in col_widths)
    content = [' | '.join(format.format(cell, width)
                          for format, cell, width in zip(row_format, row, col_widths))
               for row_format, row in zip(table_format, values)]
    formatted_header = ' | '.join('{:^{}}'.format(h, width) for h, width in zip(header, col_widths))

    return '\n'.join([formatted_header, line] + content)


class SchedulingSummary(object):
    '''
    Object representing the schedule that is used to simulate the objects in a
    network. Objects of this type are returned by `scheduling_summary`, they
    should not be created manually by the user.
    
    Parameters
    ----------
    objects : list of `BrianObject`
        The sorted list of objects that are simulated by the network.
    '''
    def __init__(self, objects):
        # Map each dt to a rank (i.e. smallest dt=0, second smallest=1, etc.)
        self.dts = dict((dt, rank) for rank, dt in
                        enumerate(sorted({float(obj.clock.dt)
                                          for obj in objects})))
        ScheduleEntry = namedtuple('ScheduleEntry',
                                   field_names=['when', 'order', 'dt',
                                                'name', 'type', 'active',
                                                'owner_name', 'owner_type'])
        self.entries = [ScheduleEntry(when=obj.when, order=obj.order,
                                      dt=obj.clock.dt, name=obj.name,
                                      type=obj.__class__.__name__,
                                      active=obj.active,
                                      owner_name=obj.group.name,
                                      owner_type=obj.group.__class__.__name__)
                        for obj in objects if not len(obj.contained_objects)]
        self.all_dts = sorted({float(entry.dt) for entry in self.entries})
        # How many steps compared to the fastest clock?
        self.steps = {float(dt): int(dt / self.all_dts[0]) for dt in self.all_dts}

    def __repr__(self):
        return _format_table(['object', 'part of', 'Clock dt', 'when', 'order', 'active'],
                             [['{} ({})'.format(entry.name, entry.type),
                               '{} ({})'.format(entry.owner_name, entry.owner_type),
                               '{} (every {})'.format(entry.dt,
                                                     'step' if self.steps[float(entry.dt)] == 1
                                                     else '{} steps'.format(self.steps[float(entry.dt)])),
                               entry.when,
                               entry.order,
                               'yes' if entry.active else 'no'] for entry in self.entries],
                             ['{:<{}}', '{:<{}}', '{:<{}}', '{:<{}}', '{:{}d}', '{:^{}}'])

    def _repr_html_(self):
        rows = ['''\
        <tr>
            <td style="text-align: left;">{}</td>
            <td style="text-align: left;">{}</td>
            <td style="text-align: left;">{}</td>
            <td style="text-align: left;">{}</td>
            <td style="text-align: right;">{}</td>
            <td style="text-align: center;">{}</td>
        </tr>
        '''.format('<b>{}</b> (<em>{}</em>)'.format(entry.name, entry.type),
                   '{} (<em>{}</em>)'.format(entry.owner_name, entry.owner_type),
                   '{} (every {})'.format(entry.dt,
                                          'step' if self.steps[float(entry.dt)] == 1
                                          else '{} steps'.format(self.steps[float(entry.dt)])),
                   entry.when,
                   entry.order,
                   'yes' if entry.active else 'no')
                for entry in self.entries]
        html_code = '''
        <table>
        <thead>
        <tr>
            <th style="text-align: center;">object</th>
            <th style="text-align: center;">part of</th>
            <th style="text-align: center;">Clock dt</th>
            <th style="text-align: center;">when</th>
            <th style="text-align: center;">order</th>
            <th style="text-align: center;">active</th>
        </tr>
        </thead>
        <tbody>
{rows}
        </tbody>
        </table>
        '''.format(rows='\n'.join(rows))
        return html_code


def _check_multiple_summed_updaters(objects):
    '''
    Helper function that checks whether multiple `SummedVariableUpdater` target
    the same target variable. Raises a `NotImplementedError` if this is the
    case (and problematic, i.e. not when using non-overlapping subgroups).

    Parameters
    ----------
    objects : list of `BrianObject`
        The list of objects in the network.
    '''
    summed_targets = {}
    for obj in objects:
        if isinstance(obj, SummedVariableUpdater):
            if obj.target_var in summed_targets:
                other_target = summed_targets[obj.target_var]
                if obj.target == other_target:
                    # We raise an error, even though this could be ok in
                    # principle (e.g. two Synapses could target different
                    # subsets of the target groups, without using subgroups)
                    msg = ('Multiple "summed variables" target the '
                           'variable "{var}" in group "{target}". Use '
                           'multiple variables in the target group '
                           'instead.'.format(var=obj.target_var.name,
                                             target=obj.target.name))
                    raise NotImplementedError(msg)
                elif (obj.target.start < other_target.stop and
                              other_target.start < obj.target.stop):
                    # Overlapping subgroups
                    msg = ('Multiple "summed variables" target the '
                           'variable "{var}" in overlapping groups '
                           '"{target1}" and "{target2}". Use separate '
                           'variables in the target groups '
                           'instead.'.format(var=obj.target_var.name,
                                             target1=other_target.name,
                                             target2=obj.target.name))
                    raise NotImplementedError(msg)
            summed_targets[obj.target_var] = obj.target


class Network(Nameable):
    '''
    Network(*objs, name='network*')
    
    The main simulation controller in Brian

    `Network` handles the running of a simulation. It contains a set of Brian
    objects that are added with `~Network.add`. The `~Network.run` method
    actually runs the simulation. The main run loop, determining which
    objects get called in what order is described in detail in the notes below.
    The objects in the `Network` are accesible via their names, e.g.
    `net['neurongroup']` would return the `NeuronGroup` with this name.
    
    Parameters
    ----------
    objs : (`BrianObject`, container), optional
        A list of objects to be added to the `Network` immediately, see
        `~Network.add`.
    name : str, optional
        An explicit name, if not specified gives an automatically generated name

    Notes
    -----
    
    The main run loop performs the following steps:
    
    1. Prepare the objects if necessary, see `~Network.prepare`.
    2. Determine the end time of the simulation as `~Network.t`+``duration``.
    3. Determine which set of clocks to update. This will be the clock with the
       smallest value of `~Clock.t`. If there are several with the same value,
       then all objects with these clocks will be updated simultaneously.
       Set `~Network.t` to the clock time.
    4. If the `~Clock.t` value of these clocks is past the end time of the
       simulation, stop running. If the `Network.stop` method or the
       `stop` function have been called, stop running. Set `~Network.t` to the
       end time of the simulation.
    5. For each object whose `~BrianObject.clock` is set to one of the clocks from the
       previous steps, call the `~BrianObject.update` method. This method will
       not be called if the `~BrianObject.active` flag is set to ``False``.
       The order in which the objects are called is described below.
    6. Increase `Clock.t` by `Clock.dt` for each of the clocks and return to
       step 2. 
    
    The order in which the objects are updated in step 4 is determined by
    the `Network.schedule` and the objects `~BrianObject.when` and
    `~BrianObject.order` attributes. The `~Network.schedule` is a list of
    string names. Each `~BrianObject.when` attribute should be one of these
    strings, and the objects will be updated in the order determined by the
    schedule. The default schedule is
    ``['start', 'groups', 'thresholds', 'synapses', 'resets', 'end']``. In
    addition to the names provided in the schedule, automatic names starting
    with ``before_`` and ``after_`` can be used. That means that all objects
    with ``when=='before_start'`` will be updated first, then
    those with ``when=='start'``, ``when=='after_start'``,
    ``when=='before_groups'``, ``when=='groups'`` and so forth. If several
    objects have the same `~BrianObject.when` attribute, then the order is
    determined by the `~BrianObject.order` attribute (lower first).
    
    See Also
    --------
    
    MagicNetwork, run, stop
    '''

    def __init__(self, *objs, **kwds):
        #: The list of objects in the Network, should not normally be modified
        #: directly.
        #: Note that in a `MagicNetwork`, this attribute only contains the
        #: objects during a run: it is filled in `before_run` and emptied in
        #: `after_run`
        self.objects = []
        
        name = kwds.pop('name', 'network*')

        if kwds:
            raise TypeError("Only keyword argument to Network is 'name'.")

        Nameable.__init__(self, name=name)

        #: Current time as a float
        self.t_ = 0.0

        for obj in objs:
            self.add(obj)

        #: Stored state of objects (store/restore)
        self._stored_state = {}

        # Stored profiling information (if activated via the keyword option)
        self._profiling_info = None

        self._schedule = None
     
    t = property(fget=lambda self: Quantity(self.t_, dim=second.dim, copy=False),
                 doc='''
                     Current simulation time in seconds (`Quantity`)
                     ''')

    @device_override('network_get_profiling_info')
    def get_profiling_info(self):
        '''
        The only reason this is not directly implemented in `profiling_info`
        is to allow devices (e.g. `CPPStandaloneDevice`) to overwrite this.
        '''
        if self._profiling_info is None:
            raise ValueError('No profiling info collected (did you run with '
                             'profile=True?)')
        return sorted(self._profiling_info, key=lambda item: item[1],
                      reverse=True)

    @property
    def profiling_info(self):
        '''
        The time spent in executing the various `CodeObject`\ s.

        A list of ``(name, time)`` tuples, containing the name of the
        `CodeObject` and the total execution time for simulations of this object
        (as a `Quantity` with unit `second`). The list is sorted descending
        with execution time.

        Profiling has to be activated using the ``profile`` keyword in `run` or
        `Network.run`.
        '''
        return self.get_profiling_info()

    _globally_stopped = False

    def __getitem__(self, item):
        if not isinstance(item, basestring):
            raise TypeError(('Need a name to access objects in a Network, '
                             'got {type} instead').format(type=type(item)))
        for obj in self.objects:
            if obj.name == item:
                return obj

        raise KeyError('No object with name "%s" found' % item)

    def __delitem__(self, key):
        if not isinstance(key, basestring):
            raise TypeError(('Need a name to access objects in a Network, '
                             'got {type} instead').format(type=type(key)))

        for obj in self.objects:
            if obj.name == key:
                self.remove(obj)
                return

        raise KeyError('No object with name "%s" found' % key)

    def __contains__(self, item):
        for obj in self.objects:
            if obj.name == item:
                return True
        return False

    def __len__(self):
        return len(self.objects)

    def __iter__(self):
        return iter(self.objects)

    def add(self, *objs):
        """
        Add objects to the `Network`
        
        Parameters
        ----------
        
        objs : (`BrianObject`, container)
            The `BrianObject` or container of Brian objects to be added. Specify
            multiple objects, or lists (or other containers) of objects.
            Containers will be added recursively. If the container is a `dict`
            then it will add the values from the dictionary but not the keys.
            If you want to add the keys, do ``add(objs.keys())``.
        """
        for obj in objs:
            if isinstance(obj, BrianObject):
                if obj._network is not None:
                    raise RuntimeError('%s has already been simulated, cannot '
                                       'add it to the network. If you were '
                                       'trying to remove and add an object to '
                                       'temporarily stop it from being run, '
                                       'set its active flag to False instead.'
                                       % obj.name)
                if obj not in self.objects:  # Don't include objects twice
                    self.objects.append(obj)
                self.add(obj.contained_objects)
            else:
                # allow adding values from dictionaries
                if isinstance(obj, Mapping):
                    self.add(*obj.values())
                else:
                    try:
                        for o in obj:
                            # The following "if" looks silly but avoids an infinite
                            # recursion if a string is provided as an argument
                            # (which might occur during testing)
                            if o is obj:
                                raise TypeError()
                            self.add(o)
                    except TypeError:
                        raise TypeError("Can only add objects of type BrianObject, "
                                        "or containers of such objects to Network")

    def remove(self, *objs):
        '''
        Remove an object or sequence of objects from a `Network`.
        
        Parameters
        ----------
        
        objs : (`BrianObject`, container)
            The `BrianObject` or container of Brian objects to be removed. Specify
            multiple objects, or lists (or other containers) of objects.
            Containers will be removed recursively.
        '''
        for obj in objs:
            if isinstance(obj, BrianObject):
                self.objects.remove(obj)
                self.remove(obj.contained_objects)
            else:
                try:
                    for o in obj:
                        self.remove(o)
                except TypeError:
                    raise TypeError("Can only remove objects of type "
                                    "BrianObject, or containers of such "
                                    "objects from Network")

    def _full_state(self):
        state = {}
        for obj in self.objects:
            if hasattr(obj, '_full_state'):
                state[obj.name] = obj._full_state()
        clocks = set([obj.clock for obj in self.objects])
        for clock in clocks:
            state[clock.name] = clock._full_state()
        # Store the time as "0_t" -- this name is guaranteed not to clash with
        # the name of an object as names are not allowed to start with a digit
        state['0_t'] = self.t_
        return state

    @device_override('network_store')
    def store(self, name='default', filename=None):
        '''
        store(name='default', filename=None)

        Store the state of the network and all included objects.

        Parameters
        ----------
        name : str, optional
            A name for the snapshot, if not specified uses ``'default'``.
        filename : str, optional
            A filename where the state should be stored. If not specified, the
            state will be stored in memory.

        Notes
        -----
        The state stored to disk can be restored with the `Network.restore`
        function. Note that it will only restore the *internal state* of all
        the objects (including undelivered spikes) -- the objects have to
        exist already and they need to have the same name as when they were
        stored. Equations, thresholds, etc. are *not* stored -- this is
        therefore not a general mechanism for object serialization. Also, the
        format of the file is not guaranteed to work across platforms or
        versions. If you are interested in storing the state of a network for
        documentation or analysis purposes use `Network.get_states` instead.
        '''
        clocks = [obj.clock for obj in self.objects]
        # Make sure that all clocks are up to date
        for clock in clocks:
            clock._set_t_update_dt(target_t=self.t)

        state = self._full_state()
        if filename is None:
            self._stored_state[name] = state
        else:
            # A single file can contain several states, so we'll read in the
            # existing file first if it exists
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    store_state = pickle.load(f)
            else:
                store_state = {}
            store_state[name] = state

            with open(filename, 'wb') as f:
                pickle.dump(store_state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @device_override('network_restore')
    def restore(self, name='default', filename=None):
        '''
        restore(name='default', filename=None)

        Retore the state of the network and all included objects.

        Parameters
        ----------
        name : str, optional
            The name of the snapshot to restore, if not specified uses
            ``'default'``.
        filename : str, optional
            The name of the file from where the state should be restored. If
            not specified, it is expected that the state exist in memory
            (i.e. `Network.store` was previously called without the ``filename``
            argument).
        '''
        if filename is None:
            state = self._stored_state[name]
        else:
            with open(filename, 'rb') as f:
                state = pickle.load(f)[name]
        self.t_ = state['0_t']
        clocks = set([obj.clock for obj in self.objects])
        restored_objects = set()
        for obj in self.objects:
            if obj.name in state:
                obj._restore_from_full_state(state[obj.name])
                restored_objects.add(obj.name)
            elif hasattr(obj, '_restore_from_full_state'):
                raise KeyError(('Stored state does not have a stored state for '
                                '"%s". Note that the names of all objects have '
                                'to be identical to the names when they were '
                                'stored.') % obj.name)
        for clock in clocks:
            clock._restore_from_full_state(state[clock.name])
        clock_names = {c.name for c in clocks}

        unnused = set(state.keys()) - restored_objects - clock_names - {'0_t'}
        if len(unnused):
            raise KeyError('The stored state contains the state of the '
                           'following objects which were not present in the '
                           'network: %s. Note that the names of all objects '
                           'have to be identical to the names when they were '
                           'stored.' % (', '.join(unnused)))

    def get_states(self, units=True, format='dict',
                   subexpressions=False, read_only_variables=True, level=0):
        '''
        Return a copy of the current state variable values of objects in the
        network.. The returned arrays are copies of the actual arrays that
        store the state variable values, therefore changing the values in the
        returned dictionary will not affect the state variables.

        Parameters
        ----------
        vars : list of str, optional
            The names of the variables to extract. If not specified, extract
            all state variables (except for internal variables, i.e. names that
            start with ``'_'``). If the ``subexpressions`` argument is ``True``,
            the current values of all subexpressions are returned as well.
        units : bool, optional
            Whether to include the physical units in the return value. Defaults
            to ``True``.
        format : str, optional
            The output format. Defaults to ``'dict'``.
        subexpressions: bool, optional
            Whether to return subexpressions when no list of variable names
            is given. Defaults to ``False``. This argument is ignored if an
            explicit list of variable names is given in ``vars``.
        read_only_variables : bool, optional
            Whether to return read-only variables (e.g. the number of neurons,
            the time, etc.). Setting it to ``False`` will assure that the
            returned state can later be used with `set_states`. Defaults to
            ``True``.
        level : int, optional
            How much higher to go up the stack to resolve external variables.
            Only relevant if extracting subexpressions that refer to external
            variables.

        Returns
        -------
        values : dict
            A dictionary mapping object names to the state variables of that
            object, in the specified ``format``.

        See Also
        --------
        VariableOwner.get_states
        '''
        states = dict()
        for obj in self.objects:
            if hasattr(obj, 'get_states'):
                states[obj.name] = obj.get_states(vars=None, units=units,
                                                  format=format,
                                                  subexpressions=subexpressions,
                                                  read_only_variables=read_only_variables,
                                                  level=level+1)
        return states

    def set_states(self, values, units=True, format='dict', level=0):
        '''
        Set the state variables of objects in the network.

        Parameters
        ----------
        values : dict
            A dictionary mapping object names to objects of ``format``, setting
            the states of this object.
        units : bool, optional
            Whether the ``values`` include physical units. Defaults to ``True``.
        format : str, optional
            The format of ``values``. Defaults to ``'dict'``
        level : int, optional
            How much higher to go up the stack to _resolve external variables.
            Only relevant when using string expressions to set values.

        See Also
        --------
        Group.set_states
        '''
        # For the moment, 'dict' is the only supported format -- later this will
        # be made into an extensible system, see github issue #306
        for obj_name, obj_values in values.iteritems():
            if obj_name not in self:
                raise KeyError(("Network does not include a network with "
                                "name '%s'.") % obj_name)
            self[obj_name].set_states(obj_values, units=units, format=format,
                                     level=level+1)


    def _get_schedule(self):
        if self._schedule is None:
            return list(prefs.core.network.default_schedule)
        else:
            return list(self._schedule)
    
    def _set_schedule(self, schedule):
        if schedule is None:
            self._schedule = None
            logger.debug('Resetting network {self.name} schedule to '
                         'default schedule')
        else:
            if (not isinstance(schedule, Sequence) or
                    not all(isinstance(slot, basestring) for slot in schedule)):
                raise TypeError('Schedule has to be None or a sequence of '
                                'scheduling slots')
            if any(slot.startswith('before_') or slot.startswith('after_')
                   for slot in schedule):
                raise ValueError('Slot names are not allowed to start with '
                                 '"before_" or "after_" -- such slot names '
                                 'are created automatically based on the '
                                 'existing slot names.')
            self._schedule = list(schedule)
            logger.debug("Setting network {self.name} schedule to "
                         "{self._schedule}".format(self=self),
                         "_set_schedule")
    
    schedule = property(fget=_get_schedule,
                        fset=_set_schedule,
                        doc='''
        List of ``when`` slots in the order they will be updated, can be modified.
        
        See notes on scheduling in `Network`. Note that additional ``when``
        slots can be added, but the schedule should contain at least all of the
        names in the default schedule:
        ``['start', 'groups', 'thresholds', 'synapses', 'resets', 'end']``.

        The schedule can also be set to ``None``, resetting it to the default
        schedule set by the `core.network.default_schedule` preference.
        ''')

    def _sort_objects(self):
        '''
        Sorts the objects in the order defined by the schedule.
        
        Objects are sorted first by their ``when`` attribute, and secondly
        by the ``order`` attribute. The order of the ``when`` attribute is
        defined by the ``schedule``. In addition to the slot names defined in
        the schedule, automatic slot names starting with ``before_`` and
        ``after_`` can be used (e.g. the slots ``['groups', 'thresholds']``
        allow to use ``['before_groups', 'groups', 'after_groups',
        'before_thresholds', 'thresholds', 'after_thresholds']`).

        Final ties are resolved using the objects' names, leading to an
        arbitrary but deterministic sorting.
        '''
        # Provided slot names are assigned positions 1, 4, 7, ...
        # before_... names are assigned positions 0, 3, 6, ...
        # after_... names are assigned positions 2, 5, 8, ...
        when_to_int = dict((when, 1+i*3)
                           for i, when in enumerate(self.schedule))
        when_to_int.update(('before_' + when, i*3)
                           for i, when in enumerate(self.schedule))
        when_to_int.update(('after_' + when, 2+i*3)
                           for i, when in enumerate(self.schedule))
        self.objects.sort(key=lambda obj: (when_to_int[obj.when],
                                           obj.order,
                                           obj.name))

    def scheduling_summary(self):
        '''
        Return a `SchedulingSummary` object, representing the scheduling
        information for all objects included in the network.
        
        Returns
        -------
        summary : `SchedulingSummary`
            Object representing the scheduling information.
        '''
        self._sort_objects()
        return SchedulingSummary(self.objects)

    def check_dependencies(self):
        all_ids = [obj.id for obj in self.objects]
        for obj in self.objects:
            for dependency in obj._dependencies:
                if not dependency in all_ids:
                    raise ValueError(('"%s" has been included in the network '
                                      'but not the object on which it '
                                      'depends.') % obj.name)

    @device_override('network_before_run')
    def before_run(self, run_namespace):
        '''
        before_run(namespace)

        Prepares the `Network` for a run.
        
        Objects in the `Network` are sorted into the correct running order, and
        their `BrianObject.before_run` methods are called.

        Parameters
        ----------
        run_namespace : dict-like, optional
            A namespace in which objects which do not define their own
            namespace will be run.
        '''
        prefs.check_all_validated()

        # Check names in the network for uniqueness
        names = [obj.name for obj in self.objects]
        non_unique_names = [name for name, count in Counter(names).iteritems()
                            if count > 1]
        if len(non_unique_names):
            formatted_names = ', '.join("'%s'" % name
                                        for name in non_unique_names)
            raise ValueError('All objects in a network need to have unique '
                             'names, the following name(s) were used more than '
                             'once: %s' % formatted_names)

        # Check that there are no SummedVariableUpdaters targeting the same
        # target variable
        _check_multiple_summed_updaters(self.objects)

        self._stopped = False
        Network._globally_stopped = False

        device = get_device()
        if device.network_schedule is not None:
            # The device defines a fixed network schedule
            if device.network_schedule != self.schedule:
                # TODO: The human-readable name of a device should be easier to get
                device_name = all_devices.keys()[all_devices.values().index(device)]
                logger.warn(("The selected device '{device_name}' only "
                             "supports a fixed schedule, but this schedule is "
                             "not consistent with the network's schedule. The "
                             "simulation will use the device's schedule.\n"
                             "Device schedule: {device.network_schedule}\n"
                             "Network schedule: {net.schedule}\n"
                             "Set the network schedule explicitly or set the "
                             "core.network.default_schedule preference to "
                             "avoid this warning.").format(device_name=device_name,
                                                           device=device,
                                                           net=self),
                            name_suffix='schedule_conflict', once=True)

        self._sort_objects()

        logger.debug("Preparing network {self.name} with {numobj} "
                     "objects: {objnames}".format(self=self,
                        numobj=len(self.objects),
                        objnames=', '.join(obj.name for obj in self.objects)),
                     "before_run")

        self.check_dependencies()

        for obj in self.objects:
            if obj.active:
                try:
                    obj.before_run(run_namespace)
                except Exception as ex:
                    raise brian_object_exception("An error occurred when preparing an object.", obj, ex)

        # Check that no object has been run as part of another network before
        for obj in self.objects:
            if obj._network is None:
                obj._network = self.id
            elif obj._network != self.id:
                raise RuntimeError(('%s has already been run in the '
                                    'context of another network. Use '
                                    'add/remove to change the objects '
                                    'in a simulated network instead of '
                                    'creating a new one.') % obj.name)

        logger.debug("Network {self.name} uses {num} "
                     "clocks: {clocknames}".format(self=self,
                        num=len(self._clocks),
                        clocknames=', '.join('%s (dt=%s)' % (obj.name, obj.dt)
                                             for obj in self._clocks)),
                     "before_run")

    @device_override('network_after_run')
    def after_run(self):
        '''
        after_run()
        '''
        for obj in self.objects:
            if obj.active:
                obj.after_run()
        
    def _nextclocks(self):
        clocks_times_dt = [(c,
                            self._clock_variables[c][1][0],
                            self._clock_variables[c][2][0])
                        for c in self._clocks]
        minclock, min_time, minclock_dt = min(clocks_times_dt, key=lambda k: k[1])
        curclocks = set(clock for clock, time, dt in clocks_times_dt if
                        (time == min_time or
                         abs(time - min_time)/min(minclock_dt, dt) < Clock.epsilon_dt))
        return minclock, curclocks

    @device_override('network_run')
    @check_units(duration=second, report_period=second)
    def run(self, duration, report=None, report_period=10*second,
            namespace=None, profile=False, level=0):
        '''
        run(duration, report=None, report_period=60*second, namespace=None, level=0)
        
        Runs the simulation for the given duration.
        
        Parameters
        ----------
        duration : `Quantity`
            The amount of simulation time to run for.
        report : {None, 'text', 'stdout', 'stderr', function}, optional
            How to report the progress of the simulation. If ``None``, do not
            report progress. If ``'text'`` or ``'stdout'`` is specified, print
            the progress to stdout. If ``'stderr'`` is specified, print the
            progress to stderr. Alternatively, you can specify a callback
            ``callable(elapsed, complete, duration)`` which will be passed
            the amount of time elapsed as a `Quantity`, the
            fraction complete from 0.0 to 1.0 and the total duration of the
            simulation (in biological time).
            The function will always be called at the beginning and the end
            (i.e. for fractions 0.0 and 1.0), regardless of the `report_period`.
        report_period : `Quantity`
            How frequently (in real time) to report progress.
        namespace : dict-like, optional
            A namespace that will be used in addition to the group-specific
            namespaces (if defined). If not specified, the locals
            and globals around the run function will be used.
        profile : bool, optional
            Whether to record profiling information (see
            `Network.profiling_info`). Defaults to ``False``.
        level : int, optional
            How deep to go up the stack frame to look for the locals/global
            (see `namespace` argument). Only used by run functions that call
            this run function, e.g. `MagicNetwork.run` to adjust for the
            additional nesting.

        Notes
        -----
        The simulation can be stopped by calling `Network.stop` or the
        global `stop` function.
        '''
        device = get_device()  # Do not use the ProxyDevice -- slightly faster
        self._clocks = set([obj.clock for obj in self.objects])
        single_clock = len(self._clocks) == 1

        t_start = self.t
        t_end = self.t + duration

        if single_clock:
            clock = list(self._clocks)[0]
            clock.set_interval(self.t, t_end)
        else:
            # We get direct references to the underlying variables for all clocks
            # to avoid expensive access during the run loop
            self._clock_variables = {c : (c.variables['timestep'].get_value(),
                                          c.variables['t'].get_value(),
                                          c.variables['dt'].get_value())
                                     for c in self._clocks}
            for clock in self._clocks:
                clock.set_interval(self.t, t_end)

        # Get the local namespace
        if namespace is None:
            namespace = get_local_namespace(level=level+3)

        self.before_run(namespace)

        if len(self.objects)==0:
            return  # TODO: raise an error? warning?

        start_time = time.time()

        logger.debug("Simulating network '%s' from time %s to %s." % (self.name,
                                                                      t_start,
                                                                      t_end),
                     'run')

        if report is not None:
            report_period = float(report_period)
            next_report_time = start_time + report_period
            if report == 'text' or report == 'stdout':
                report_callback = TextReport(sys.stdout)
            elif report == 'stderr':
                report_callback = TextReport(sys.stderr)
            elif isinstance(report, basestring):
                raise ValueError(('Do not know how to handle report argument '
                                  '"%s".' % report))
            elif callable(report):
                report_callback = report
            else:
                raise TypeError(('Do not know how to handle report argument, '
                                 'it has to be one of "text", "stdout", '
                                 '"stderr", or a callable function/object, '
                                 'but it is of type %s') % type(report))
            report_callback(0*second, 0.0, t_start, duration)

        profiling_info = defaultdict(float)

        if single_clock:
            timestep, t, dt = (clock.variables['timestep'].get_value(),
                               clock.variables['t'].get_value(),
                               clock.variables['dt'].get_value())
        else:
            # Find the first clock to be updated (see note below)
            clock, curclocks = self._nextclocks()
            timestep, _, _ = self._clock_variables[clock]

        running = timestep[0] < clock._i_end

        active_objects = [obj for obj in self.objects if obj.active]

        while running and not self._stopped and not Network._globally_stopped:
            if not single_clock:
                timestep, t, dt = self._clock_variables[clock]
            # update the network time to this clock's time
            self.t_ = t[0]
            if report is not None:
                current = time.time()
                if current > next_report_time:
                    report_callback((current-start_time)*second,
                                    (self.t_ - float(t_start))/float(t_end),
                                    t_start, duration)
                    next_report_time = current + report_period

            # update the objects and tick forward the clock(s)
            if single_clock:
                if profile:
                    for obj in active_objects:
                        obj_time = time.time()
                        obj.run()
                        profiling_info[obj.name] += (time.time() - obj_time)
                else:
                    for obj in active_objects:
                        obj.run()

                timestep[0] += 1
                t[0] = timestep[0] * dt[0]
            else:
                if profile:
                    for obj in active_objects:
                        if obj._clock in curclocks:
                            obj_time = time.time()
                            obj.run()
                            profiling_info[obj.name] += (time.time() - obj_time)
                else:
                    for obj in active_objects:
                        if obj._clock in curclocks:
                            obj.run()

                for c in curclocks:
                    timestep, t, dt = self._clock_variables[c]
                    timestep[0] += 1
                    t[0] = timestep[0] * dt[0]
                # find the next clocks to be updated. The < operator for Clock
                # determines that the first clock to be updated should be the one
                # with the smallest t value, unless there are several with the
                # same t value in which case we update all of them
                clock, curclocks = self._nextclocks()
                timestep, _, _ = self._clock_variables[clock]

            if device._maximum_run_time is not None and time.time()-start_time>float(device._maximum_run_time):
                self._stopped = True
            else:
                running = timestep[0] < clock._i_end

        end_time = time.time()
        if self._stopped or Network._globally_stopped:
            self.t_ = clock.t_
        else:
            self.t_ = float(t_end)

        device._last_run_time = end_time-start_time
        if duration>0:
            device._last_run_completed_fraction = (self.t-t_start)/duration
        else:
            device._last_run_completed_fraction = 1.0

        # check for nans
        for obj in self.objects:
            if isinstance(obj, Group):
                obj._check_for_invalid_states()

        if report is not None:
            report_callback((end_time-start_time)*second, 1.0, t_start, duration)
        self.after_run()

        logger.debug(("Finished simulating network '%s' "
                      "(took %.2fs)") % (self.name, end_time-start_time),
                     'run')
        # Store profiling info (or erase old info to avoid confusion)
        if profile:
            self._profiling_info = [(name, t*second)
                                    for name, t in profiling_info.iteritems()]
            # Dump a profiling summary to the log
            logger.debug('\n' + str(profiling_summary(self)))
        else:
            self._profiling_info = None

    @device_override('network_stop')
    def stop(self):
        '''
        stop()

        Stops the network from running, this is reset the next time `Network.run` is called.
        '''
        self._stopped = True

    def __repr__(self):
        return '<%s at time t=%s, containing objects: %s>' % (self.__class__.__name__,
                                                              str(self.t),
                                                              ', '.join((obj.__repr__() for obj in self.objects)))


class ProfilingSummary(object):
    '''
    Class to nicely display the results of profiling. Objects of this class are
    returned by `profiling_summary`.

    Parameters
    ----------

    net : `Network`
        The `Network` object to profile.
    show : int, optional
        The number of results to show (the longest results will be shown). If
        not specified, all results will be shown.

    See Also
    --------
    Network.profiling_info
    '''
    def __init__(self, net, show=None):
        prof = net.profiling_info
        if len(prof):
            names, times = zip(*prof)
        else:  # Can happen if a network has been run for 0ms
            # Use a dummy entry to prevent problems with empty lists later
            names = ['no code objects have been run']
            times = [0*second]
        self.total_time = sum(times)
        self.time_unit = msecond
        if self.total_time>1*second:
            self.time_unit = second
        if show is not None:
            names = names[:show]
            times = times[:show]
        if self.total_time>0*second:
            self.percentages = [100.0*time/self.total_time for time in times]
        else:
            self.percentages = [0. for _ in times]
        self.names_maxlen = max(len(name) for name in names)
        self.names = [name+' '*(self.names_maxlen-len(name)) for name in names]
        self.times = times

    def __repr__(self):
        times = ['%.2f %s' % (time/self.time_unit, self.time_unit) for time in self.times]
        times_maxlen = max(len(time) for time in times)
        times = [' '*(times_maxlen-len(time))+time for time in times]
        percentages = ['%.2f %%' % percentage for percentage in self.percentages]
        percentages_maxlen = max(len(percentage) for percentage in percentages)
        percentages = [(' '*(percentages_maxlen-len(percentage)))+percentage for percentage in percentages]

        s = 'Profiling summary'
        s += '\n'+'='*len(s)+'\n'
        for name, time, percentage in zip(self.names, times, percentages):
            s += '%s    %s    %s\n' % (name, time, percentage)
        return s

    def _repr_html_(self):
        times = ['%.2f %s' % (time/self.time_unit, self.time_unit) for time in self.times]
        percentages = ['%.2f %%' % percentage for percentage in self.percentages]
        s = '<h2 class="brian_prof_summary_header">Profiling summary</h2>\n'
        s += '<table class="brian_prof_summary_table">\n'
        for name, time, percentage in zip(self.names, times, percentages):
            s += '<tr>'
            s += '<td>%s</td>' % name
            s += '<td style="text-align: right">%s</td>' % time
            s += '<td style="text-align: right">%s</td>' % percentage
            s += '</tr>\n'
        s += '</table>'
        return s


def profiling_summary(net=None, show=None):
    '''
    Returns a `ProfilingSummary` of the profiling info for a run. This object
    can be transformed to a string explicitly but on an interactive console
    simply calling `profiling_summary` is enough since it will
    automatically convert the `ProfilingSummary` object.

    Parameters
    ----------

    net : {`Network`, None} optional
        The `Network` object to profile, or `magic_network` if not specified.
    show : int
        The number of results to show (the longest results will be shown). If
        not specified, all results will be shown.
    '''
    if net is None:
        from .magic import magic_network
        net = magic_network
    return ProfilingSummary(net, show)


def scheduling_summary(net=None):
    '''
    Returns a `SchedulingSummary` object, representing the scheduling
    information for all objects included in the given `Network` (or the
    "magic" network, if none is specified). The returned objects can be
    printed or converted to a string to give an ASCII table representation of
    the schedule. In a Jupyter notebook, the output can be displayed as a
    HTML table.

    Parameters
    ----------
    net : `Network`, optional
        The network for which the scheduling information should be displayed.
        Defaults to the "magic" network.

    Returns
    -------
    summary : `SchedulingSummary`
        An object that represents the scheduling information.
    '''
    if net is None:
        from .magic import magic_network
        magic_network._update_magic_objects(level=1)
        net = magic_network
    return net.scheduling_summary()


def schedule_propagation_offset(net=None):
    '''
    Returns the minimal time difference for a post-synaptic effect after a
    spike. With the default schedule, this time difference is 0, since the
    ``thresholds`` slot precedes the ``synapses`` slot. For the GeNN device,
    however, a post-synaptic effect will occur in the following time step, this
    function therefore returns one ``dt``.

    Parameters
    ----------
    net : `Network`
        The network to check (uses the magic network if not specified).

    Returns
    -------
    offset : `Quantity`
        The minimum spike propagation delay: ``0*ms`` for the standard schedule
        but ``dt`` for schedules where ``synapses`` precedes ``thresholds``.

    Notes
    -----
    This function always returns ``0*ms`` or ``defaultclock.dt`` -- no attempt
    is made to deal with other clocks.
    '''
    from brian2.core.magic import magic_network

    device = get_device()
    if device.network_schedule is not None:
        schedule = device.network_schedule
    else:
        if net is None:
            net = magic_network
        schedule = net.schedule

    if schedule.index('thresholds') < schedule.index('synapses'):
        return 0*second
    else:
        return defaultclock.dt
