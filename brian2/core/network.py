'''
Module defining the `Network` object, the basis of all simulation runs.

Preferences
-----------

.. document_brian_prefs:: core.network

'''

import sys
import gc
import time
from collections import defaultdict, Sequence

from brian2.utils.logger import get_logger
from brian2.core.names import Nameable
from brian2.core.base import BrianObject
from brian2.core.clocks import Clock
from brian2.units.fundamentalunits import check_units, DimensionMismatchError
from brian2.units.allunits import second, msecond 
from brian2.core.preferences import prefs, BrianPreference

from .base import device_override

__all__ = ['Network', 'profiling_summary']


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

    def __call__(self, elapsed, completed, duration):
        if completed == 0.0:
            self.stream.write(('Starting simulation for duration '
                               '%s\n') % duration)
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

        #: Stored time for the store/restore mechanism
        self._stored_t = {}

        # Stored profiling information (if activated via the keyword option)
        self._profiling_info = None

        self._schedule = None
     
    t = property(fget=lambda self: self.t_*second,
                 doc='''
                     Current simulation time in seconds (`Quantity`)
                     ''')

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
        if self._profiling_info is None:
            raise ValueError('(No profiling info collected (did you run with '
                             'profile=True?)')
        return sorted(self._profiling_info, key=lambda item: item[1],
                      reverse=True)

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
            Containers will be added recursively.
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
                self.objects.append(obj)
                self.add(obj.contained_objects)
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

    @device_override('network_store')
    def store(self, name='default'):
        '''
        store(name='default')

        Store the state of the network and all included objects.

        Parameters
        ----------
        name : str, optional
            A name for the snapshot, if not specified uses ``'default'``.

        '''
        self._stored_t[name] = self.t_
        clocks = [obj.clock for obj in self.objects]
        # Make sure that all clocks are up to date
        for clock in clocks:
            clock._set_t_update_dt(t=self.t)

        for obj in self.objects:
            if hasattr(obj, '_store'):
                obj._store(name=name)

    @device_override('network_restore')
    def restore(self, name='default'):
        '''
        restore(name='default')

        Retore the state of the network and all included objects.

        Parameters
        ----------
        name : str, optional
            The name of the snapshot to restore, if not specified uses
            ``'default'``.

        '''
        for obj in self.objects:
            if hasattr(obj, '_restore'):
                obj._restore(name=name)
        self.t_ = self._stored_t[name]

    def _get_schedule(self):
        if self._schedule is None:
            return list(prefs.core.network.default_schedule)
        else:
            return list(self._schedule)
    
    def _set_schedule(self, schedule):
        if schedule is None:
            self._schedule = None
            logger.debug('Reset network {self.name} schedule to '
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
            logger.debug("Set network {self.name} schedule to "
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

    def check_dependencies(self):
        all_ids = [obj.id for obj in self.objects]
        for obj in self.objects:
            for dependency in obj._dependencies:
                if not dependency in all_ids:
                    raise ValueError(('"%s" has been included in the network '
                                      'but not the object on which it '
                                      'depends.') % obj.name)

    @device_override('network_before_run')
    def before_run(self, run_namespace=None, level=0):
        '''
        before_run(namespace)

        Prepares the `Network` for a run.
        
        Objects in the `Network` are sorted into the correct running order, and
        their `BrianObject.before_run` methods are called.

        Parameters
        ----------
        namespace : dict-like, optional
            A namespace in which objects which do not define their own
            namespace will be run.
        '''
        from brian2.devices.device import get_device, all_devices

        # A garbage collection here can be useful to free memory if we have
        # multiple runs
        gc.collect()

        prefs.check_all_validated()
        
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
                    obj.before_run(run_namespace, level=level+2)
                except DimensionMismatchError as ex:
                    raise DimensionMismatchError(('An error occured preparing '
                                                  'object "%s":\n%s') % (obj.name,
                                                                          ex.desc),
                                                 *ex.dims)

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

        logger.debug("Network {self.name} has {num} "
                     "clocks: {clocknames}".format(self=self,
                        num=len(self._clocks),
                        clocknames=', '.join(obj.name for obj in self._clocks)),
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
        # Getting Clock.t_ is relatively expensive since it involves a
        # multiplication therefore we extract it only once
        clocks_times = [(clock, clock.t_) for clock in self._clocks]
        minclock, min_time = min(clocks_times, key=lambda k: k[1])
        curclocks = set(clock for clock, time in clocks_times if
                        (time == min_time or
                         abs(time - min_time)<Clock.epsilon))
        return minclock, curclocks

    @device_override('network_run')
    @check_units(duration=second, report_period=second)
    def run(self, duration, report=None, report_period=10*second,
            namespace=None, profile=True, level=0):
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
            `Network.profiling_info`). Defaults to ``True``.
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
        self._clocks = set([obj.clock for obj in self.objects])
        t_start = self.t
        t_end = self.t+duration
        for clock in self._clocks:
            clock.set_interval(self.t, t_end)

        self.before_run(namespace, level=level+3)

        if len(self.objects)==0:
            return # TODO: raise an error? warning?

        # Find the first clock to be updated (see note below)
        clock, curclocks = self._nextclocks()
        if report is not None:
            report_period = float(report_period)
            start = current = time.time()
            next_report_time = start + report_period
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
            report_callback(0*second, 0.0, duration)

        profiling_info = defaultdict(float)

        while clock.running and not self._stopped and not Network._globally_stopped:
            # update the network time to this clocks time
            self.t_ = clock.t_
            if report is not None:
                current = time.time()
                if current > next_report_time:
                    report_callback((current-start)*second,
                                    (self.t_ - float(t_start))/float(t_end),
                                    duration)
                    next_report_time = current + report_period
                # update the objects with this clock
            for obj in self.objects:
                if obj._clock in curclocks and obj.active:
                    if profile:
                        obj_time = time.time()
                        obj.run()
                        profiling_info[obj.name] += (time.time() - obj_time)
                    else:
                        obj.run()

            # tick the clock forward one time step
            for c in curclocks:
                c.tick()
            # find the next clocks to be updated. The < operator for Clock
            # determines that the first clock to be updated should be the one
            # with the smallest t value, unless there are several with the 
            # same t value in which case we update all of them
            clock, curclocks = self._nextclocks()

        if self._stopped or Network._globally_stopped:
            self.t_ = clock.t_
        else:
            self.t_ = float(t_end)

        if report is not None:
            report_callback((current-start)*second, 1.0, duration)
        self.after_run()

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
