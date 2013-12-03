'''
Module providing the `Synapses` class and related helper classes/functions.
'''

import collections
from collections import defaultdict
import weakref
import itertools
import re

import numpy as np

from brian2.core.namespace import create_namespace
from brian2.core.preferences import brian_prefs
from brian2.core.variables import (ArrayVariable, Variable,
                                   Subexpression, AttributeVariable,
                                   StochasticVariable)
from brian2.devices.device import get_device
from brian2.equations.equations import (Equations, SingleEquation,
                                        DIFFERENTIAL_EQUATION, STATIC_EQUATION,
                                        PARAMETER)
from brian2.groups.group import Group, GroupCodeRunner, create_runner_codeobj
from brian2.stateupdaters.base import StateUpdateMethod
from brian2.stateupdaters.exact import independent
from brian2.units.fundamentalunits import (Unit, Quantity,
                                           fail_for_dimension_mismatch)
from brian2.units.allunits import second
from brian2.utils.logger import get_logger
from brian2.core.namespace import get_local_namespace
from brian2.core.spikesource import SpikeSource


MAX_SYNAPSES = 2147483647

__all__ = ['Synapses']

logger = get_logger(__name__)


class StateUpdater(GroupCodeRunner):
    '''
    The `GroupCodeRunner` that updates the state variables of a `Synapses`
    at every timestep.
    '''
    def __init__(self, group, method):
        self.method_choice = method
        GroupCodeRunner.__init__(self, group,
                                 'stateupdate',
                                 when=(group.clock, 'groups'),
                                 name=group.name + '_stateupdater',
                                 check_units=False)

        self.method = StateUpdateMethod.determine_stateupdater(self.group.equations,
                                                               self.group.variables,
                                                               method)
    
    def update_abstract_code(self):
        
        self.method = StateUpdateMethod.determine_stateupdater(self.group.equations,
                                                               self.group.variables,
                                                               self.method_choice)
        
        self.abstract_code = self.method(self.group.equations,
                                         self.group.variables)


class SummedVariableUpdater(GroupCodeRunner):
    '''
    The `GroupCodeRunner` that updates a value in the target group with the
    sum over values in the `Synapses` object.
    '''
    def __init__(self, expression, target_varname, synapses, target):

        # Handling sumped variables using the standard mechanisms is not
        # possible, we therefore also directly give the names of the arrays
        # to the template.

        code = '''
        _synaptic_var = {expression}
        {target_varname} = {target_varname}
        '''.format(expression=expression,
                   target_varname=target_varname)

        template_kwds = {'_target_var_array': synapses.variables[target_varname].arrayname}

        GroupCodeRunner.__init__(self, group=synapses,
                                 template='summed_variable',
                                 code=code,
                                 # We want to update the sumned variable before
                                 # the target group gets updated
                                 when=(target.clock, 'groups', -1),
                                 name=synapses.name + '_summed_variable_' + target_varname,
                                 template_kwds=template_kwds)


class SynapticPathway(GroupCodeRunner, Group):
    '''
    The `GroupCodeRunner` that applies the pre/post statement(s) to the state
    variables of synapses where the pre-/postsynaptic group spiked in this
    time step.

    Parameters
    ----------

    synapses : `Synapses`
        Reference to the main `Synapses` object
    prepost : {'pre', 'post'}
        Whether this object should react to pre- or postsynaptic spikes
    objname : str, optional
        The name to use for the object, will be appendend to the name of
        `synapses` to create a name in the sense of `Nameable`. The `synapses`
        object should allow access to this object via
        ``synapses.getattr(objname)``. It has to use the actual `objname`
        attribute instead of relying on the provided argument, since the name
        may have changed to become unique. If ``None`` is provided (the
        default), ``prepost+'*'`` will be used (see `Nameable` for an
        explanation of the wildcard operator).
    '''
    def __init__(self, synapses, code, prepost, objname=None):
        self.code = code
        self.prepost = prepost
        if prepost == 'pre':
            self.source = synapses.source
            self.target = synapses.target
            self.synapse_sources = synapses.variables['_synaptic_pre']
        elif prepost == 'post':
            self.source = synapses.target
            self.target = synapses.source
            self.synapse_sources = synapses.variables['_synaptic_post']
        else:
            raise ValueError('prepost argument has to be either "pre" or '
                             '"post"')
        self.synapses = synapses

        if objname is None:
            objname = prepost + '*'

        GroupCodeRunner.__init__(self, synapses,
                                 'synapses',
                                 code=code,
                                 when=(synapses.clock, 'synapses'),
                                 name=synapses.name + '_' + objname,
                                 template_kwds={'pathway': self})

        self._pushspikes_codeobj = None

        self.spikes_start = self.source.start
        self.spikes_stop = self.source.stop

        self.spiking_synapses = []
        self.variables = {'_spiking_synapses': AttributeVariable(Unit(1),
                                                                  self,
                                                                  'spiking_synapses',
                                                                  constant=False),
                           'delay': get_device().dynamic_array_1d(self, 'delay',
                                                                  synapses._N,
                                                                  second,
                                                                  constant=True)}
        self._delays = self.variables['delay']
        # Register the object with the `SynapticIndex` object so it gets
        # automatically resized
        synapses.register_variable(self._delays)

        # Re-extract the last part of the name from the full name
        self.objname = self.name[len(synapses.name) + 1:]

        #: The simulation dt (necessary for the delays)
        self.dt = self.synapses.clock.dt_

        #: The `SpikeQueue`
        self.queue = None

        #: The `CodeObject` initalising the `SpikeQueue` at the begin of a run
        self._initialise_queue_codeobj = None

        self.namespace = create_namespace(None)
        # Enable access to the delay attribute via the specifier
        self._enable_group_attributes()

    def update_abstract_code(self):
        if self.synapses.event_driven is not None:
            event_driven_update = independent(self.synapses.event_driven,
                                              self.group.variables)
            # TODO: Any way to do this more elegantly?
            event_driven_update = re.sub(r'\bdt\b', '(t - lastupdate)',
                                         event_driven_update)

            self.abstract_code = event_driven_update + '\n'
        else:
            self.abstract_code = ''

        self.abstract_code += self.code + '\n'
        self.abstract_code += 'lastupdate = t\n'

    def before_run(self, namespace):
        # execute code to initalize the spike queue
        if self._initialise_queue_codeobj is None:
            self._initialise_queue_codeobj = create_runner_codeobj(self,
                                                                   '', # no code,
                                                                   'synapses_initialise_queue',
                                                                   name=self.name+'_initialise_queue',
                                                                   check_units=False,
                                                                   additional_variables=self.group.variables)
        self._initialise_queue_codeobj()
        GroupCodeRunner.before_run(self, namespace)

        # we insert rather than replace because GroupCodeRunner puts a CodeObject in updaters already
        if self._pushspikes_codeobj is None:
            self._pushspikes_codeobj = create_runner_codeobj(self,
                                                             '', # no code
                                                             'synapses_push_spikes',
                                                             name=self.name+'_push_spikes',
                                                             check_units=False,
                                                             additional_variables=self.group.variables)

        self.updaters.insert(0, self._pushspikes_codeobj.get_updater())

    def initialise_queue(self):
        if self.queue is None:
            self.queue = get_device().spike_queue(self.source.start, self.source.stop)

        # Update the dt (might have changed between runs)
        self.dt = self.synapses.clock.dt_

        self.queue.prepare(self._delays.get_value(), self.dt,
                           self.synapse_sources.get_value())

    def push_spikes(self):
        # Push new spikes into the queue
        spikes = self.source.spikes
        if len(spikes):
            self.queue.push(spikes)
        # Get the spikes
        self.spiking_synapses = self.queue.peek()
        # Advance the spike queue
        self.queue.advance()


def slice_to_test(x):
    '''
    Returns a testing function corresponding to whether an index is in slice x.
    x can also be an int.
    '''
    if isinstance(x,int):
        return lambda y: (y == x)
    elif isinstance(x, slice):
        if isinstance(x, slice) and x == slice(None):
            # No need for testing
            return lambda y: np.repeat(True, len(y))
        start, stop, step = x.start, x.stop, x.step

        if start is None:
            # No need to test for >= start
            if step is None:
                # Only have a stop value
                return lambda y: (y < stop)
            else:
                # Stop and step
                return lambda y: (y < stop) & ((y % step) == 0)
        else:
            # We need to test for >= start
            if step is None:
                if stop is None:
                    # Only a start value
                    return lambda y: (y >= start)
                else:
                    # Start and stop
                    return lambda y: (y >= start) & (y < stop)
            else:
                if stop is None:
                    # Start and step value
                    return lambda y: (y >= start) & ((y-start)%step == 0)
                else:
                    # Start, step and stop
                    return lambda y: (y >= start) & ((y-start)%step == 0) & (y < stop)
    else:
        raise TypeError('Expected int or slice, got {} instead'.format(type(x)))


def find_synapses(index, synaptic_neuron):
    if isinstance(index, (int, slice)):
        test = slice_to_test(index)
        found = test(synaptic_neuron)
        synapses = np.flatnonzero(found)
    else:
        synapses = []
        for neuron in index:
            targets = np.flatnonzero(synaptic_neuron == neuron)
            synapses.extend(targets)

    return synapses


def _synapse_numbers(pre_neurons, post_neurons):
    # Build an array of synapse numbers by counting the number of times
    # a source/target combination exists
    synapse_numbers = np.zeros_like(pre_neurons)
    numbers = {}
    for i, (source, target) in enumerate(zip(pre_neurons,
                                             post_neurons)):
        number = numbers.get((source, target), 0)
        synapse_numbers[i] = number
        numbers[(source, target)] = number + 1
    return synapse_numbers


class Synapses(Group):
    '''
    Class representing synaptic connections. Creating a new `Synapses` object
    does by default not create any synapses -- you either have to provide
    the `connect` argument or call the `Synapses.connect` method for that.

    Parameters
    ----------

    source : `SpikeSource`
        The source of spikes, e.g. a `NeuronGroup`.
    target : `Group`, optional
        The target of the spikes, typically a `NeuronGroup`. If none is given,
        the same as `source`
    model : {`str`, `Equations`}, optional
        The model equations for the synapses.
    pre : {str, dict}, optional
        The code that will be executed after every pre-synaptic spike. Can be
        either a single (possibly multi-line) string, or a dictionary mapping
        pathway names to code strings. In the first case, the pathway will be
        called ``pre`` and made available as an attribute of the same name.
        In the latter case, the given names will be used as the
        pathway/attribute names. Each pathway has its own code and its own
        delays.
    post : {str, dict}, optional
        The code that will be executed after every post-synaptic spike. Same
        conventions as for `pre`, the default name for the pathway is ``post``.
    connect : {str, bool}. optional
        Determines whether any actual synapses are created. ``False`` (the
        default) means not to create any synapses, ``True`` means to create
        synapses between all source/target pairs. Also accepts a string
        expression that evaluates to ``True`` for every synapse that should
        be created, e.g. ``'i == j'`` for a one-to-one connectivity. See
        `Synapses.connect` for more details.
    delay : {`Quantity`, dict}, optional
        The delay for the "pre" pathway (same for all synapses) or a dictionary
        mapping pathway names to delays. If a delay is specified in this way
        for a pathway, it is stored as a single scalar value. It can still
        be changed afterwards, but only to a single scalar value. If you want
        to have delays that vary across synapses, do not use the keyword
        argument, but instead set the delays via the attribute of the pathway,
        e.g. ``S.pre.delay = ...`` (or ``S.delay = ...`` as an abbreviation),
        ``S.post.delay = ...``, etc.
    namespace : dict, optional
        A dictionary mapping identifier names to objects. If not given, the
        namespace will be filled in at the time of the call of `Network.run`,
        with either the values from the ``network`` argument of the
        `Network.run` method or from the local context, if no such argument is
        given.
    dtype : `dtype`, optional
        The standard datatype for all state variables. Defaults to
        `core.default_scalar_type`.
    codeobj_class : class, optional
        The `CodeObject` class to use to run code.
    clock : `Clock`, optional
        The clock to use.
    method : {str, `StateUpdateMethod`}, optional
        The numerical integration method to use. If none is given, an
        appropriate one is automatically determined.
    name : str, optional
        The name for this object. If none is given, a unique name of the form
        ``synapses``, ``synapses_1``, etc. will be automatically chosen.
    '''
    def __init__(self, source, target=None, model=None, pre=None, post=None,
                 connect=False, delay=None, namespace=None, dtype=None,
                 codeobj_class=None,
                 clock=None, method=None, name='synapses*'):
        self._N = 0
        Group.__init__(self, when=clock, name=name)
        
        self.codeobj_class = codeobj_class

        self.source = weakref.proxy(source)
        if target is None:
            self.target = self.source
        else:
            self.target = weakref.proxy(target)
            
        ##### Prepare and validate equations
        if model is None:
            model = ''

        if isinstance(model, basestring):
            model = Equations(model)
        if not isinstance(model, Equations):
            raise TypeError(('model has to be a string or an Equations '
                             'object, is "%s" instead.') % type(model))

        # Check flags
        model.check_flags({DIFFERENTIAL_EQUATION: ['event-driven'],
                           STATIC_EQUATION: ['summed'],
                           PARAMETER: ['constant']})

        # Separate the equations into event-driven and continuously updated
        # equations
        event_driven = []
        continuous = []
        for single_equation in model.itervalues():
            if 'event-driven' in single_equation.flags:
                event_driven.append(single_equation)
            else:
                continuous.append(single_equation)
        # Add the lastupdate variable, used by event-driven equations
        continuous.append(SingleEquation(PARAMETER, 'lastupdate', second))

        if len(event_driven):
            self.event_driven = Equations(event_driven)
        else:
            self.event_driven = None

        self.equations = Equations(continuous)

        # Setup the namespace
        self._given_namespace = namespace
        self.namespace = create_namespace(namespace)

        self._queues = {}
        self._delays = {}

        # Setup variables
        self.variables = self._create_variables()

        #: List of `Variable` objects that should be resized when the number of
        #: synapses changes
        self._registered_variables = []

        for var in self.variables.itervalues():
            #if isinstance(var, DynamicArrayVariable):
            if hasattr(var, 'resize'):
                # Register the array with the `SynapticItemMapping` object so
                # it gets automatically resized
                self.register_variable(var)

        #: List of names of all updaters, e.g. ['pre', 'post']
        self._synaptic_updaters = []
        #: List of all `SynapticPathway` objects
        self._pathways = []
        for prepost, argument in zip(('pre', 'post'), (pre, post)):
            if not argument:
                continue
            if isinstance(argument, basestring):
                self._add_updater(argument, prepost)
            elif isinstance(argument, collections.Mapping):
                for key, value in argument.iteritems():
                    if not isinstance(key, basestring):
                        err_msg = ('Keys for the "{}" argument'
                                   'have to be strings, got '
                                   '{} instead.').format(prepost, type(key))
                        raise TypeError(err_msg)
                    self._add_updater(value, prepost, objname=key)

        # If we have a pathway called "pre" (the most common use case), provide
        # direct access to its delay via a delay attribute (instead of having
        # to use pre.delay)
        if 'pre' in self._synaptic_updaters:
            self.variables['delay'] = self.pre.variables['delay']

        if delay is not None:
            if isinstance(delay, Quantity):
                if not 'pre' in self._synaptic_updaters:
                    raise ValueError(('Cannot set delay, no "pre" pathway exists.'
                                      'Use a dictionary if you want to set the '
                                      'delay for a pathway with a different name.'))
                delay = {'pre': delay}

            if not isinstance(delay, collections.Mapping):
                raise TypeError('Delay argument has to be a quantity or a '
                                'dictionary, is type %s instead.' % type(delay))
            for pathway, pathway_delay in delay.iteritems():
                if not pathway in self._synaptic_updaters:
                    raise ValueError(('Cannot set the delay for pathway '
                                      '"%s": unknown pathway.') % pathway)
                if not isinstance(pathway_delay, Quantity):
                    raise TypeError(('Cannot set the delay for pathway "%s": '
                                     'expected a quantity, got %s instead.') % (pathway,
                                                                                type(pathway_delay)))
                if pathway_delay.size != 1:
                    raise TypeError(('Cannot set the delay for pathway "%s": '
                                     'expected a scalar quantity, got a '
                                     'quantity with shape %s instead.') % str(pathway_delay.shape))
                fail_for_dimension_mismatch(pathway_delay, second, ('Delay has to be '
                                                                    'specified in units '
                                                                    'of seconds'))
                updater = getattr(self, pathway)
                # For simplicity, store the delay as a one-element array
                # so that for example updater._delays[:] works.
                updater._delays.resize(1)
                updater._delays[0] = float(pathway_delay)
                updater._delays.scalar = True
                # Do not resize the scalar delay variable when adding synapses
                self.unregister_variable(updater._delays)

        #: Performs numerical integration step
        self.state_updater = StateUpdater(self, method)        
        self.contained_objects.append(self.state_updater)

        #: "Summed variable" mechanism -- sum over all synapses of a
        #: pre-/postsynaptic target
        self.summed_updaters = {}
        # We want to raise an error if the same variable is updated twice
        # using this mechanism. This could happen if the Synapses object
        # connected a NeuronGroup to itself since then all variables are
        # accessible as var_pre and var_post.
        summed_targets = set()
        for single_equation in self.equations.itervalues():
            if 'summed' in single_equation.flags:
                varname = single_equation.varname
                if not (varname.endswith('_pre') or varname.endswith('_post')):
                    raise ValueError(('The summed variable "%s" does not end '
                                      'in "_pre" or "_post".') % varname)
                if not varname in self.variables:
                    raise ValueError(('The summed variable "%s" does not refer'
                                      'do any known variable in the '
                                      'target group.') % varname)
                if varname.endswith('_pre'):
                    summed_target = self.source
                    orig_varname = varname[:-4]
                else:
                    summed_target = self.target
                    orig_varname = varname[:-5]

                target_eq = getattr(summed_target, 'equations', {}).get(orig_varname, None)
                if target_eq is None or target_eq.type != PARAMETER:
                    raise ValueError(('The summed variable "%s" needs a '
                                      'corresponding parameter "%s" in the '
                                      'target group.') % (varname,
                                                          orig_varname))

                fail_for_dimension_mismatch(self.variables['_summed_'+varname].unit,
                                            self.variables[varname].unit,
                                            ('Summed variables need to have '
                                             'the same units in Synapses '
                                             'and the target group'))
                if self.variables[varname] in summed_targets:
                    raise ValueError(('The target variable "%s" is already '
                                      'updated by another summed '
                                      'variable') % orig_varname)
                summed_targets.add(self.variables[varname])
                updater = SummedVariableUpdater(single_equation.expr,
                                                varname, self, summed_target)
                self.summed_updaters[varname] = updater
                self.contained_objects.append(updater)

        # Do an initial connect, if requested
        if not isinstance(connect, (bool, basestring)):
            raise TypeError(('"connect" keyword has to be a boolean value or a '
                             'string, is type %s instead.' % type(connect)))
        self._initial_connect = connect
        if not connect is False:
            self.connect(connect, level=1)

        # Activate name attribute access
        self._enable_group_attributes()

    def __len__(self):
        return self._N

    def before_run(self, namespace):
        self.lastupdate = self.clock.t
        super(Synapses, self).before_run(namespace)

    def _add_updater(self, code, prepost, objname=None):
        '''
        Add a new target updater. Users should call `add_pre` or `add_post`
        instead.

        Parameters
        ----------
        code : str
            The abstract code that should be executed on pre-/postsynaptic
            spikes.
        prepost : {'pre', 'post'}
            Whether the code is triggered by presynaptic or postsynaptic spikes
        objname : str, optional
            A name for the object, see `SynapticPathway` for more details.

        Returns
        -------
        objname : str
            The final name for the object. Equals `objname` if it was explicitly
            given (and did not end in a wildcard character).

        '''
        if prepost == 'pre':
            spike_group, group_name = self.source, 'Source'
        elif prepost == 'post':
            spike_group, group_name = self.target, 'Target'
        else:
            raise ValueError(('"prepost" argument has to be "pre" or "post", '
                              'is "%s".') % prepost)

        if not isinstance(spike_group, SpikeSource) or not hasattr(spike_group, 'clock'):
            raise TypeError(('%s has to be a SpikeSource with spikes and'
                             ' clock attribute. Is type %r instead')
                            % (group_name, type(spike_group)))

        updater = SynapticPathway(self, code, prepost, objname)
        objname = updater.objname
        if hasattr(self, objname):
            raise ValueError(('Cannot add updater with name "{name}", synapses '
                              'object already has an attribute with this '
                              'name.').format(name=objname))

        setattr(self, objname, updater)
        self._synaptic_updaters.append(objname)
        self._pathways.append(updater)
        self.contained_objects.append(updater)
        return objname

    def _create_variables(self, dtype=None):
        '''
        Create the variables dictionary for this `Synapses`, containing
        entries for the equation variables and some standard entries.
        '''
        if dtype is None:
            dtype = defaultdict(lambda: brian_prefs['core.default_scalar_dtype'])
        elif isinstance(dtype, np.dtype):
            dtype = defaultdict(lambda: dtype)
        elif not hasattr(dtype, '__getitem__'):
            raise TypeError(('Cannot use type %s as dtype '
                             'specification') % type(dtype))

        v = {}

        # Add all the pre and post variables with _pre and _post suffixes
        self.variable_indices = defaultdict(lambda: '_idx')
        for name, var in getattr(self.source, 'variables', {}).iteritems():
            v[name + '_pre'] = var
            if isinstance(var, ArrayVariable):
                self.variable_indices[name + '_pre'] = '_presynaptic_idx'
        for name, var in getattr(self.target, 'variables', {}).iteritems():
            v[name + '_post'] = var
            if isinstance(var, ArrayVariable):
                self.variable_indices[name + '_post'] = '_postsynaptic_idx'
            # Also add all the post variables without a suffix -- if this
            # clashes with the name of a state variable defined in this
            # Synapses group, the latter will overwrite the entry later and
            # take precedence
            v[name] = var
            if isinstance(var, ArrayVariable):
                self.variable_indices[name] = '_postsynaptic_idx'

        dev = get_device()
        # Standard variables always present
        v.update({'_num_source_neurons': Variable(Unit(1), len(self.source),
                                                  constant=True),
                  '_num_target_neurons': Variable(Unit(1), len(self.target),
                                                  constant=True),
                  '_synaptic_pre': dev.dynamic_array_1d(self, '_synaptic_pre',
                                                        0, Unit(1), dtype=np.int32,
                                                        constant_size=True),
                  '_synaptic_post': dev.dynamic_array_1d(self, '_synaptic_post',
                                                         0, Unit(1), dtype=np.int32,
                                                         constant_size=True)})

        # Allow accessing the pre- and postsynaptic indices in a nicer way
        v.update({'i': self.source.variables['i'],
                  'j': self.target.variables['i'],
                  # we have to make a distinction here between the indices
                  # and the arrays (even though they refer to the same object)
                  # the synaptic propagation template would otherwise overwrite
                  # synaptic_post in its namespace with the value of the
                  # postsynaptic index, leading to errors for the next
                  # propagation
                  '_presynaptic_idx': v['_synaptic_pre'],
                  '_postsynaptic_idx': v['_synaptic_post']
                  })
        self.variable_indices['i'] = '_presynaptic_idx'
        self.variable_indices['j'] = '_postsynaptic_idx'

        # Add the standard variables (this also overwrites their inherited
        # values from the postsynaptic group)
        v.update(Group._create_variables(self))

        # For synapses, the number of synapses is not fixed (as for other
        # groups), therefore replace the "N" entry
        v['N'] = AttributeVariable(Unit(1), self, '_N', constant=True,
                                   read_only=True)

        for eq in itertools.chain(self.equations.itervalues(),
                                  self.event_driven.itervalues()
                                  if self.event_driven is not None else []):
            if eq.type in (DIFFERENTIAL_EQUATION, PARAMETER):
                constant = ('constant' in eq.flags)
                # We are dealing with dynamic arrays here, code generation
                # shouldn't directly access the specifier.array attribute but
                # use specifier.get_value() to get a reference to the underlying
                # array
                v[eq.varname] = dev.dynamic_array_1d(self,
                                                     eq.varname,
                                                     0,
                                                     eq.unit,
                                                     dtype=dtype[eq.varname],
                                                     constant=constant,
                                                     is_bool=eq.is_bool)
                if eq.varname in self.variable_indices:
                    # we are overwriting a postsynaptic variable of the same
                    # name, delete the reference to the postsynaptic index
                    del self.variable_indices[eq.varname]
            elif eq.type == STATIC_EQUATION:
                if 'summed' in eq.flags:
                    # Give a special name to the subexpression for summed
                    # variables to avoid confusion with the pre/postsynaptic
                    # target variable
                    varname = '_summed_'+eq.varname
                else:
                    varname = eq.varname
                v.update({varname: Subexpression(varname,
                                                 eq.unit,
                                                 dtype=brian_prefs['core.default_scalar_dtype'],
                                                 expr=str(eq.expr),
                                                 group=self,
                                                 is_bool=eq.is_bool)})
            else:
                raise AssertionError('Unknown type of equation: ' + eq.eq_type)

        # Stochastic variables
        for xi in self.equations.stochastic_variables:
            v.update({xi: StochasticVariable()})

        return v

    def connect(self, pre_or_cond, post=None, p=1., n=1, level=0):
        '''
        Add synapses. The first argument can be either a presynaptic index
        (int or array) or a condition for synapse creation in the form of a
        string that evaluates to a boolean value (or directly a boolean value).
        If it is given as an index, also `post` has to be present. A string
        condition will be evaluated for all pre-/postsynaptic indices, which
        can be referred to as `i` and `j`.

        Parameters
        ----------
        pre_or_cond : {int, ndarray of int, bool, str}
            The presynaptic neurons (in the form of an index or an array of
            indices) or a boolean value or a string that evaluates to a
            boolean value. If it is an index, then also `post` has to be
            given.
        post_neurons : {int, ndarray of int), optional
            GroupIndices of neurons from the target group. Non-optional if one or
            more presynaptic indices have been given.
        p : float, optional
            The probability to create `n` synapses wherever the condition
            given as `pre_or_cond` evaluates to true or for the given
            pre/post indices.
        n : int, optional
            The number of synapses to create per pre/post connection pair.
            Defaults to 1.

        Examples
        --------
        >>> from brian2 import *
        >>> import numpy as np
        >>> G = NeuronGroup(10, 'dv/dt = -v / tau : 1', threshold='v>1', reset='v=0')
        >>> S = Synapses(G, G, 'w:1', pre='v+=w')
        >>> S.connect('i != j') # all-to-all but no self-connections
        >>> S.connect(0, 0) # connect neuron 0 to itself
        >>> S.connect(np.array([1, 2]), np.array([2, 1])) # connect 1->2 and 2->1
        >>> S.connect(True) # connect all-to-all
        >>> S.connect('i != j', p=0.1)  # Connect neurons with 10% probability, exclude self-connections
        >>> S.connect('i == j', n=2)  # Connect all neurons to themselves with 2 synapses
        '''
        if not isinstance(pre_or_cond, (bool, basestring)):
            pre_or_cond = np.asarray(pre_or_cond)
            if not np.issubdtype(pre_or_cond.dtype, np.int):
                raise TypeError(('Presynaptic indices have to be given as '
                                 'integers, are type %s instead.') % pre_or_cond.dtype)

            post = np.asarray(post)
            if not np.issubdtype(post.dtype, np.int):
                raise TypeError(('Presynaptic indices can only be combined '
                                 'with postsynaptic integer indices))'))
            if isinstance(n, basestring):
                raise TypeError(('Indices cannot be combined with a string'
                                 'expression for n. Either use an array/scalar '
                                 'for n, or a string expression for the '
                                 'connections'))
            i, j, n = np.broadcast_arrays(pre_or_cond, post, n)
            if i.ndim > 1:
                raise ValueError('Can only use 1-dimensional indices')
            self._add_synapses(i, j, n, p, level=level+1)
        elif isinstance(pre_or_cond, (basestring, bool)):
            if pre_or_cond is False:
                return  # nothing to do...
            elif pre_or_cond is True:
                # TODO: This should not be handled with the general mechanism
                pre_or_cond = 'True'
            if post is not None:
                raise ValueError('Cannot give a postsynaptic index when '
                                 'using a string expression')
            if not isinstance(n, (int, basestring)):
                raise TypeError('n has to be an integer or a string evaluating '
                                'to an integer, is type %s instead.' % type(n))
            if not isinstance(p, (float, basestring)):
                raise TypeError('p has to be a float or a string evaluating '
                                'to an float, is type %s instead.' % type(n))
            self._add_synapses(None, None, n, p, condition=pre_or_cond,
                               level=level+1)
        else:
            raise TypeError(('First argument has to be an index or a '
                             'string, is %s instead.') % type(pre_or_cond))

    def _resize(self, number):
        if not isinstance(number, int):
            raise TypeError(('Expected an integer number got {} '
                             'instead').format(type(number)))
        if number < self._N:
            raise ValueError(('Cannot reduce number of synapses, '
                              '{} < {}').format(number, len(self)))

        for variable in self._registered_variables:
            variable.resize(number)

        self._N = number

    def register_variable(self, variable):
        '''
        Register a `DynamicArray` to be automatically resized when the size of
        the indices change. Called automatically when a `SynapticArrayVariable`
        specifier is created.
        '''
        if not hasattr(variable, 'resize'):
            raise TypeError(('Variable of type {} does not have a resize '
                             'method, cannot register it with the synaptic '
                             'indices.').format(type(variable)))
        self._registered_variables.append(weakref.proxy(variable))

    def unregister_variable(self, variable):
        '''
        Unregister a `DynamicArray` from the automatic resizing mechanism.
        '''
        proxy_var = weakref.proxy(variable)
        # The same variable might have been registered more than once
        while proxy_var in self._registered_variables:
            self._registered_variables.remove(proxy_var)

    def _add_synapses(self, sources, targets, n, p, condition=None,
                      level=0):

        if condition is None:
            sources = np.atleast_1d(sources).astype(np.int32)
            targets = np.atleast_1d(targets).astype(np.int32)
            n = np.atleast_1d(n)
            p = np.atleast_1d(p)
            if not len(p) == 1 or p != 1:
                use_connections = np.random.rand(len(sources)) < p
                sources = sources[use_connections]
                targets = targets[use_connections]
                n = n[use_connections]
            sources = sources.repeat(n)
            targets = targets.repeat(n)
            new_synapses = len(sources)

            old_N = len(self)
            new_N = old_N + new_synapses
            self._resize(new_N)

            # Deal with subgroups
            if '_sub_idx' in self.source.variables:
                real_sources = self.source.variables['_sub_idx'][sources]
            else:
                real_sources = sources
            if '_sub_idx' in self.target.variables:
                real_targets = self.target.variables['_sub_idx'][targets]
            else:
                real_targets = targets
            self.variables['_synaptic_pre'].get_value()[old_N:new_N] = real_sources
            self.variables['_synaptic_post'].get_value()[old_N:new_N] = real_targets
            synapse_idx = old_N
        else:
            abstract_code = '_pre_idcs = _all_pre \n'
            abstract_code += '_post_idcs = _all_post \n'
            abstract_code += '_cond = ' + condition + '\n'
            abstract_code += '_n = ' + str(n) + '\n'
            abstract_code += '_p = ' + str(p)
            namespace = get_local_namespace(level + 1)
            additional_namespace = ('implicit-namespace', namespace)
            # This overwrites 'i' and 'j' in the synapses' variables dictionary
            # This is necessary because in the context of synapse creation, i
            # and j do not correspond to the sources/targets of the existing
            # synapses but to all the possible sources/targets
            variables = {
                # Will be set in the template
                'i': Variable(unit=Unit(1), constant=True),
                'j': Variable(unit=Unit(1), constant=True),
            }
            if '_sub_idx' in self.source.variables:
                variables['_all_pre'] = self.source.variables['_sub_idx']
            else:
                variables['_all_pre'] = self.source.variables['i']

            if '_sub_idx' in self.target.variables:
                variables['_all_post'] = self.target.variables['_sub_idx']
            else:
                variables['_all_post'] = self.target.variables['i']

            variable_indices = defaultdict(lambda: '_idx')
            for varname in self.variables:
                if self.variable_indices[varname] == '_presynaptic_idx':
                    variable_indices[varname] = '_all_pre'
                elif self.variable_indices[varname] == '_postsynaptic_idx':
                    variable_indices[varname] = '_all_post'
            variable_indices['_all_pre'] = 'i'
            variable_indices['_all_post'] = 'j'
            codeobj = create_runner_codeobj(self,
                                            abstract_code,
                                            'synapses_create',
                                            variable_indices=variable_indices,
                                            additional_variables=variables,
                                            additional_namespace=additional_namespace,
                                            check_units=False
                                            )
            codeobj()
        number = len(self.variables['_synaptic_pre'])
        self._resize(number)

    def calc_indices(self, index):
        '''
        Returns synaptic indices for `index`, which can be a tuple of indices
        (including arrays and slices), a single index or a string.

        '''
        if (not isinstance(index, (tuple, basestring)) and
                isinstance(index, (int, np.ndarray, slice,
                                   collections.Sequence))):
            index = (index, slice(None), slice(None))
        if isinstance(index, tuple):
            if len(index) == 2:  # two indices (pre- and postsynaptic cell)
                index = (index[0], index[1], slice(None))
            elif len(index) > 3:
                raise IndexError('Need 1, 2 or 3 indices, got %d.' % len(index))

            I, J, K = index

            pre_synapses = find_synapses(I, self.variables['_synaptic_pre'].get_value() - self.source.start)
            post_synapses = find_synapses(J, self.variables['_synaptic_post'].get_value() - self.target.start)
            matching_synapses = np.intersect1d(pre_synapses, post_synapses,
                                               assume_unique=True)

            if isinstance(K, slice) and K == slice(None):
                return matching_synapses
            elif isinstance(K, (int, slice)):
                test_k = slice_to_test(K)
            else:
                raise NotImplementedError(('Indexing synapses with arrays not'
                                           'implemented yet'))

            # We want to access the raw arrays here, not go through the Variable
            pre_neurons = self.variables['_synaptic_pre'].get_value()[pre_synapses]
            post_neurons = self.variables['_synaptic_post'].get_value()[post_synapses]
            synapse_numbers = _synapse_numbers(pre_neurons,
                                               post_neurons)
            return np.intersect1d(matching_synapses,
                                  np.flatnonzero(test_k(synapse_numbers)),
                                  assume_unique=True)
        else:
            raise IndexError('Unsupported index type {itype}'.format(itype=type(index)))

