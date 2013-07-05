import collections
import weakref
import itertools
import inspect
import re

import numpy as np

from brian2.core.base import BrianObject
from brian2.core.namespace import create_namespace
from brian2.core.preferences import brian_prefs
from brian2.core.specifiers import (ArrayVariable, Index, DynamicArrayVariable, 
                                    AttributeValue, Subexpression, ReadOnlyValue,
                                    StochasticVariable, SynapticArrayVariable,
                                    Specifier)
from brian2.codegen.languages import PythonLanguage
from brian2.equations.equations import (Equations, SingleEquation,
                                        DIFFERENTIAL_EQUATION, STATIC_EQUATION,
                                        PARAMETER)
from brian2.groups.group import Group, GroupCodeRunner, create_codeobj
from brian2.memory.dynamicarray import DynamicArray1D
from brian2.stateupdaters.base import StateUpdateMethod
from brian2.stateupdaters.exact import independent
from brian2.units.fundamentalunits import Unit, Quantity, fail_for_dimension_mismatch, get_dimensions
from brian2.units.allunits import second
from brian2.utils.logger import get_logger
from brian2.utils.stringtools import get_identifiers
from brian2.core.namespace import get_local_namespace

from .spikequeue import SpikeQueue

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
        indices = {'_neuron_idx': Index('_neuron_idx', True)}
        GroupCodeRunner.__init__(self, group,
                                 group.language.template_state_update,
                                 indices=indices,
                                 when=(group.clock, 'groups'),
                                 name=group.name + '_stateupdater',
                                 check_units=False)

        self.method = StateUpdateMethod.determine_stateupdater(self.group.equations,
                                                               self.group.specifiers,
                                                               method)
    
    def update_abstract_code(self):
        
        self.method = StateUpdateMethod.determine_stateupdater(self.group.equations,
                                                               self.group.specifiers,
                                                               self.method_choice)
        
        self.abstract_code = self.method(self.group.equations,
                                         self.group.specifiers)


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
        if prepost == 'pre':
            self.source = synapses.source
            self.synapse_indices = synapses.indices.pre_synaptic
        elif prepost == 'post':
            self.source = synapses.target
            self.synapse_indices = synapses.indices.post_synaptic
        else:
            raise ValueError('prepost argument has to be either "pre" or '
                             '"post"')
        self.synapses = synapses
        indices = {'_neuron_idx': Index('_neuron_idx', False),
                   '_postsynaptic_idx': Index('_postsynaptic_idx', False),
                   '_presynaptic_idx': Index('_presynaptic_idx', False)}
        self._delays = DynamicArray1D(len(synapses.indices), dtype=np.float64)
        self.queue = SpikeQueue()
        self.spiking_synapses = []
        self.specifiers = {'_spiking_synapses': AttributeValue('_spiking_synapses',
                                                               Unit(1), np.int,
                                                               self, 'spiking_synapses'),
                           'delay': SynapticArrayVariable('delay', second,
                                                          np.float64,
                                                          self._delays,
                                                          '_neuron_idx',
                                                          self.synapses)}
        if objname is None:
            objname = prepost + '*'

        GroupCodeRunner.__init__(self, synapses,
                                 synapses.language.template_synapses,
                                 code=code,
                                 indices=indices,
                                 when=(synapses.clock, 'synapses'),
                                 name=synapses.name + '_' + objname)

        # Re-extract the last part of the name from the full name
        self.objname = self.name[len(synapses.name) + 1:]

        #: The simulation dt (necessary for the delays)
        self.dt = self.synapses.clock.dt_

        # Enable access to the delay attribute via the specifier
        Group.__init__(self)

    def update_abstract_code(self):
        if self.synapses.event_driven is not None:
            event_driven_update = independent(self.synapses.event_driven,
                                              self.group.specifiers)
            # TODO: Any way to do this more elegantly?
            event_driven_update = re.sub(r'\bdt\b', '(t - lastupdate)',
                                         event_driven_update)

            self.abstract_code = event_driven_update + '\n'
        else:
            self.abstract_code = ''

        self.abstract_code += self.code + '\n'
        self.abstract_code += 'lastupdate = t\n'

    def pre_run(self, namespace):
        # Update the dt (might have changed between runs)
        self.dt = self.synapses.clock.dt_
        GroupCodeRunner.pre_run(self, namespace)
        self.queue.compress(np.round(self._delays[:] / self.dt).astype(np.int),
                            self.synapse_indices, len(self.synapses))
    
    def pre_update(self):
        # Push new spikes into the queue
        spikes = self.source.spikes
        if len(spikes):
            indices = np.hstack((self.synapse_indices[spike]
                                 for spike in spikes)).astype(np.int32)
            if len(indices):
                if len(self._delays) > 1:
                    delays = np.round(self._delays[indices] / self.dt).astype(int)
                else:
                    delays = np.round(self._delays / self.dt).astype(int)
                self.queue.push(indices, delays)
        # Get the spikes
        self.spiking_synapses = self.queue.peek()
        # Advance the spike queue
        self.queue.next()


class IndexView(object):

    def __init__(self, indices, mapping):
        self.indices = indices
        self.mapping = mapping

    def __getitem__(self, item):
        synaptic_indices = self.indices[item]
        return self.mapping[synaptic_indices]


class SynapseIndexView(object):

    def __init__(self, indices):
        self.indices = indices

    def __getitem__(self, item):
        pre = self.indices.i[item]
        post = self.indices.j[item]

        return _synapse_numbers(pre, post)


def slice_to_test(x):
    '''
    Returns a testing function corresponding to whether an index is in slice x.
    x can also be an int.
    '''
    if isinstance(x,int):
        return lambda y: (y == x)
    elif isinstance(x, slice):
        if x == slice(None):
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


def find_synapses(index, neuron_synaptic, synaptic_neuron):
    if isinstance(index, (int, slice)):
        test = slice_to_test(index)
        neurons = test(synaptic_neuron[:])
        synapses = np.flatnonzero(neurons)
    else:
        neurons = []
        synapses = []
        for neuron in index:
            targets = neuron_synaptic[neuron]
            neurons.extend([neuron] * len(targets))
            synapses.extend(targets)

    return neurons, synapses


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


class SynapticIndices(object):
    '''
    Convenience object to store the synaptic indices.

    Parameters
    ----------
    synapses : `Synapses`
        Reference to the main `Synapses object`
    '''
    def __init__(self, synapses):
        self.source_len = len(synapses.source)
        self.target_len = len(synapses.target)
        self.synapses = weakref.proxy(synapses)
        dtype = smallest_inttype(MAX_SYNAPSES)
        self.synaptic_pre = DynamicArray1D(0, dtype=dtype)
        self.synaptic_post = DynamicArray1D(0, dtype=dtype)
        self.pre_synaptic = [DynamicArray1D(0, dtype=dtype)
                             for _ in xrange(self.source_len)]
        self.post_synaptic = [DynamicArray1D(0, dtype=dtype)
                              for _ in xrange(self.target_len)]
        self.i = IndexView(self, self.synaptic_pre)
        self.j = IndexView(self, self.synaptic_post)
        self.k = SynapseIndexView(self)

        self.specifiers = self.specifiers = {'i': DynamicArrayVariable('i',
                                                                       Unit(1),
                                                                       self.synaptic_pre.dtype,
                                                                       self.synaptic_pre,
                                                                       '_neuron_idx'),
                                             'j': DynamicArrayVariable('j',
                                                                       Unit(1),
                                                                       self.synaptic_post.dtype,
                                                                       self.synaptic_post,
                                                                       '_neuron_idx')}

        self._registered_variables = []

    N = property(fget=lambda self: len(self.synaptic_pre),
                 doc='Total number of synapses')

    def _resize(self, number):
        if not isinstance(number, int):
            raise TypeError(('Expected an integer number got {} '
                             'instead').format(type(number)))
        if number < self.N:
            raise ValueError(('Cannot reduce number of synapses, '
                              '{} < {}').format(number, self.N))

        self.synaptic_pre.resize(number)
        self.synaptic_post.resize(number)

        for variable in self._registered_variables:
            variable.resize(number)

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
            if not np.isscalar(p) or p != 1:
                use_connections = np.random.rand(len(sources)) < p
                sources = sources[use_connections]
                targets = targets[use_connections]
                n = n[use_connections]
            sources = sources.repeat(n)
            targets = targets.repeat(n)
            new_synapses = len(sources)

            old_N = self.N
            new_N = old_N + new_synapses
            self._resize(new_N)

            self.synaptic_pre[old_N:new_N] = sources
            self.synaptic_post[old_N:new_N] = targets
            synapse_idx = old_N
            for source, target in zip(sources, targets):
                synapses = self.pre_synaptic[source]
                synapses.resize(len(synapses) + 1)
                synapses[-1] = synapse_idx
                synapses = self.post_synaptic[target]
                synapses.resize(len(synapses) + 1)
                synapses[-1] = synapse_idx
                synapse_idx += 1
        else:
            abstract_code = '_cond = ' + condition + '\n'
            abstract_code += '_n = ' + str(n) + '\n'
            abstract_code += '_p = ' + str(p)
            namespace = get_local_namespace(level + 1)
            additional_namespace = ('implicit-namespace', namespace)
            specifiers = {
                '_num_source_neurons': ReadOnlyValue('_num_source_neurons', Unit(1),
                                                     np.int32, self.source_len),
                '_num_target_neurons': ReadOnlyValue('_num_target_neurons', Unit(1),
                                                     np.int32, self.target_len),
                '_synaptic_pre': ReadOnlyValue('_synaptic_pre', Unit(1),
                                               np.int32,
                                               self.synaptic_pre),
                '_synaptic_post': ReadOnlyValue('_synaptic_post', Unit(1),
                                                np.int32,
                                                self.synaptic_post),
                '_pre_synaptic': ReadOnlyValue('_pre_synaptic', Unit(1),
                                               np.int32,
                                               self.pre_synaptic),
                '_post_synaptic': ReadOnlyValue('_post_synaptic', Unit(1),
                                                np.int32,
                                                self.post_synaptic),
                # Will be set in the template
                'i': Specifier('i'),
                'j': Specifier('j')
            }
            codeobj = create_codeobj(self.synapses,
                                     abstract_code,
                                     self.synapses.language.template_synapses_create,
                                     {},
                                     additional_specifiers=specifiers,
                                     additional_namespace=additional_namespace,
                                     check_units=False,
                                     language=self.synapses.language)
            codeobj()
            number = len(self.synaptic_pre)
            for variable in self._registered_variables:
                variable.resize(number)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
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

            pre_neurons, pre_synapses = find_synapses(I, self.pre_synaptic,
                                                      self.synaptic_pre)
            post_neurons, post_synapses = find_synapses(J, self.post_synaptic,
                                                        self.synaptic_post)

            matching_synapses = np.intersect1d(pre_synapses, post_synapses,
                                               assume_unique=True)

            if K == slice(None):
                return matching_synapses
            elif isinstance(K, (int, slice)):
                test_k = slice_to_test(K)
            else:
                raise NotImplementedError(('Indexing synapses with arrays not'
                                           'implemented yet'))

            synapse_numbers = _synapse_numbers(pre_neurons,
                                               post_neurons)

            return np.intersect1d(matching_synapses,
                                  np.flatnonzero(test_k(synapse_numbers)),
                                  assume_unique=True)

        elif isinstance(index, basestring):
            # interpret the string expression
            identifiers = get_identifiers(index)
            specifiers = dict(self.specifiers)
            if 'k' in identifiers:
                synapse_numbers = _synapse_numbers(self.synaptic_pre[:],
                                                   self.synaptic_post[:])
                specifiers['k'] = ArrayVariable('k', Unit(1), np.int32,
                                                synapse_numbers,
                                                '_neuron_idx')
            # Get the locals and globals from the stack frame
            frame = inspect.stack()[2][0]
            namespace = dict(frame.f_globals)
            namespace.update(frame.f_locals)
            additional_namespace = ('implicit-namespace', namespace)
            indices = {'_neuron_idx': Index('_neuron_idx', iterate_all=True)}
            abstract_code = '_cond = ' + index
            codeobj = create_codeobj(self.synapses,
                                     abstract_code,
                                     self.synapses.language.template_state_variable_indexing,
                                     indices,
                                     additional_specifiers=specifiers,
                                     additional_namespace=additional_namespace,
                                     check_units=False,
                                     language=self.synapses.language)

            result = codeobj()
            return result
        else:
            raise IndexError('Unsupported index type {}'.format(type(index)))


class Synapses(BrianObject, Group):
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
    equations : {`str`, `Equations`}, optional
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
    language : `Language`, optional
        The code-generation language to use. Defaults to `PythonLanguage`.
    clock : `Clock`, optional
        The clock to use.
    method : {str, `StateUpdateMethod`}, optional
        The numerical integration method to use. If none is given, an
        appropriate one is automatically determined.
    name : str, optional
        The name for this object. If none is given, a unique name of the form
        ``synapses``, ``synapses_1``, etc. will be automatically chosen.
    '''
    def __init__(self, source, target=None, equations=None, pre=None, post=None,
                 connect=False, delay=None, namespace=None, dtype=None,
                 language=None, clock=None, method=None, name='synapses*'):
        
        BrianObject.__init__(self, when=clock, name=name)

        if not hasattr(source, 'spikes') and hasattr(source, 'clock'):
            raise TypeError(('Source has to be a SpikeSource with spikes and'
                             ' clock attribute. Is type %r instead')
                            % type(source))

        self.source = weakref.proxy(source)
        if target is None:
            self.target = self.source
        else:
            self.target = weakref.proxy(target)
            
        ##### Prepare and validate equations
        if isinstance(equations, basestring):
            equations = Equations(equations)
        if not isinstance(equations, Equations):
            raise TypeError(('equations has to be a string or an Equations '
                             'object, is "%s" instead.') % type(equations))

        # Check flags
        equations.check_flags({DIFFERENTIAL_EQUATION: ('event-driven'),
                               PARAMETER: ('constant')})

        # Separate the equations into event-driven and continuously updated
        # equations
        event_driven = []
        continuous = []
        for single_equation in equations.itervalues():
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

        ##### Setup the memory
        self.arrays = self._allocate_memory(dtype=dtype)

        # Setup the namespace
        self._given_namespace = namespace
        self.namespace = create_namespace(0, namespace)

        # Code generation (TODO: this should be refactored and modularised)
        # Temporary, set default language to Python
        if language is None:
            language = PythonLanguage()
        self.language = language
        
        self._queues = {}
        self._delays = {}

        self.indices = SynapticIndices(self)
        # Allow S.i instead of S.indices.i, etc.
        self.i = self.indices.i
        self.j = self.indices.j
        self.k = self.indices.k

        # Setup specifiers
        self.specifiers = self._create_specifiers()

        #: List of names of all updaters, e.g. ['pre', 'post']
        self._updaters = []
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
        if 'pre' in self._updaters:
            self.specifiers['delay'] = SynapticArrayVariable('delay', second,
                                                             np.float64,
                                                             self.pre._delays,
                                                             '_neuron_idx',
                                                             self)

        if delay is not None:
            if isinstance(delay, Quantity):
                if not 'pre' in self._updaters:
                    raise ValueError(('Cannot set delay, no "pre" pathway exists.'
                                      'Use a dictionary if you want to set the '
                                      'delay for a pathway with a different name.'))
                delay = {'pre': delay}

            if not isinstance(delay, collections.Mapping):
                raise TypeError('Delay argument has to be a quantity or a '
                                'dictionary, is type %s instead.' % type(delay))
            for pathway, pathway_delay in delay.iteritems():
                if not pathway in self._updaters:
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
                self.indices.unregister_variable(updater._delays)
                del updater._delays
                # For simplicity, store the delay as a one-element array
                # so that for example updater._delays[:] works.
                updater._delays = np.array([float(pathway_delay)])
                specifier = ArrayVariable('delay', second, np.float64,
                                          updater._delays, None,
                                          group=updater)
                updater.specifiers['delay'] = specifier
                if pathway == 'pre':
                    self.specifiers['delay'] = specifier

        #: Performs numerical integration step
        self.state_updater = StateUpdater(self, method)        
        self.contained_objects.append(self.state_updater)

        # Do an initial connect, if requested
        if not isinstance(connect, (bool, basestring)):
            raise TypeError(('"connect" keyword has to be a boolean value or a '
                             'string, is type %s instead.' % type(connect)))
        self._initial_connect = connect
        if not connect is False:
            self.connect(connect, level=1)

        # Activate name attribute access
        Group.__init__(self)

    N = property(fget=lambda self: self.indices.N,
                 doc='Total number of synapses')

    def __len__(self):
        return self.N

    def pre_run(self, namespace):
        self.lastupdate = self.clock.t
        super(Synapses, self).pre_run(namespace)

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
        updater = SynapticPathway(self, code, prepost, objname)
        objname = updater.objname
        if hasattr(self, objname):
            raise ValueError(('Cannot add updater with name "{name}", synapses '
                              'object already has an attribute with this '
                              'name.').format(name=objname))

        setattr(self, objname, updater)
        self._updaters.append(objname)
        self.contained_objects.append(updater)
        return objname

    def _create_specifiers(self):
        '''
        Create the specifiers dictionary for this `NeuronGroup`, containing
        entries for the equation variables and some standard entries.
        '''
        # Add all the pre and post specifiers with _pre and _post suffixes
        s = {}
        for name, spec in getattr(self.source, 'specifiers', {}).iteritems():
            if isinstance(spec, ArrayVariable):
                new_spec = ArrayVariable(spec.name, spec.unit, spec.dtype,
                                         spec.array, '_presynaptic_idx',
                                         self)
                s[name + '_pre'] = new_spec
        for name, spec in getattr(self.target, 'specifiers', {}).iteritems():
            if isinstance(spec, ArrayVariable):
                new_spec = ArrayVariable(spec.name, spec.unit, spec.dtype,
                             spec.array, '_postsynaptic_idx', self)
                s[name + '_post'] = new_spec
                # Also add all the post specifiers without a suffix -- if this
                # clashes with the name of a state variable defined in this
                # Synapses group, the latter will overwrite the entry later and
                # take precedence
                s[name] = new_spec

        # Standard specifiers always present
        s.update({'t': AttributeValue('t',  second, np.float64,
                                      self.clock, 't_'),
                  'dt': AttributeValue('dt', second, np.float64,
                                       self.clock, 'dt_', constant=True),
                  '_num_neurons': AttributeValue('_num_neurons', Unit(1),
                                                 np.int, self, 'N',
                                                 constant=True),
                  # We don't need "proper" specifier for these -- they are not accessed in user code
                  '_synaptic_pre': ReadOnlyValue('_synaptic_pre', Unit(1),
                                                 np.int32,
                                                 self.indices.synaptic_pre),
                  '_synaptic_post': ReadOnlyValue('_synaptic_post', Unit(1),
                                                  np.int32,
                                                  self.indices.synaptic_post),
                  '_pre_synaptic': ReadOnlyValue('_pre_synaptic', Unit(1),
                                                 np.int32,
                                                 self.indices.pre_synaptic),
                  '_post_synaptic': ReadOnlyValue('_post_synaptic', Unit(1),
                                                  np.int32,
                                                  self.indices.post_synaptic)})

        for eq in itertools.chain(self.equations.itervalues(),
                                  self.event_driven.itervalues() if self.event_driven is not None else []):
            if eq.type in (DIFFERENTIAL_EQUATION, PARAMETER):
                array = self.arrays[eq.varname]
                constant = ('constant' in eq.flags)
                # We are dealing with dynamic arrays here, code generation
                # shouldn't directly access the specifier.array attribute but
                # use specifier.get_value() to get a reference to the underlying
                # array
                s.update({eq.varname: SynapticArrayVariable(eq.varname,
                                                    eq.unit,
                                                    array.dtype,
                                                    array,
                                                    '_neuron_idx',
                                                    self,
                                                    constant=constant)})
        
            elif eq.type == STATIC_EQUATION:
                s.update({eq.varname: Subexpression(eq.varname, eq.unit,
                                                    brian_prefs['core.default_scalar_dtype'],
                                                    str(eq.expr),
                                                    s,
                                                    self.namespace)})
            else:
                raise AssertionError('Unknown type of equation: ' + eq.eq_type)

        # Stochastic variables
        for xi in self.equations.stochastic_variables:
            s.update({xi: StochasticVariable(xi)})

        return s

    def _allocate_memory(self, dtype=None):
        # Allocate memory (TODO: this should be refactored somewhere at some point)
        arrayvarnames = set(eq.varname for eq in self.equations.itervalues() if
                            eq.type in (DIFFERENTIAL_EQUATION,
                                           PARAMETER))
        if self.event_driven is not None:
            # Only differential equations are event-driven
            arrayvarnames |= set(eq.varname
                                 for eq in self.event_driven.itervalues())

        arrays = {}
        for name in arrayvarnames:
            if isinstance(dtype, dict):
                curdtype = dtype[name]
            else:
                curdtype = dtype
            if curdtype is None:
                curdtype = brian_prefs['core.default_scalar_dtype']
            arrays[name] = DynamicArray1D(0)
        logger.debug("NeuronGroup memory allocated successfully.")
        return arrays             

    def connect_one_to_one(self):
        ''' Manually create a one to one connectivity pattern '''

        if len(self.source) != len(self.target):
            raise TypeError('Can only create synapses between groups of same size')

        self.connect(np.arange(len(self.source)),
                     np.arange(len(self.target)))

    def connect_full(self):
        '''
        Connect all neurons in the source group to all neurons in the target
        group.
        '''
        sources, targets = np.meshgrid(np.arange(len(self.source)),
                                       np.arange(len(self.target)))
        self.connect(sources.flat(), targets.flat())

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
            Indices of neurons from the target group. Non-optional if one or
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
        if (not isinstance(pre_or_cond, bool) and
                isinstance(pre_or_cond, (int, np.ndarray))):
            if not isinstance(post, (int, np.ndarray)):
                raise TypeError(('Presynaptic indices can only be combined '
                                 'with postsynaptic indices))'))
            if isinstance(n, basestring):
                raise TypeError(('Indices cannot be combined with a string'
                                 'expression for n. Either use an array/scalar '
                                 'for n, or a string expression for the '
                                 'connections'))
            i, j, n = np.broadcast_arrays(pre_or_cond, post, n)
            if i.ndim > 1:
                raise ValueError('Can only use 1-dimensional indices')

            self.indices._add_synapses(i, j, n, p, level=level+1)
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
            self.indices._add_synapses(None, None, n, p, condition=pre_or_cond,
                                       level=level+1)
        else:
            raise TypeError(('First argument has to be an index or a '
                             'string, is %s instead.') % type(pre_or_cond))


def smallest_inttype(N):
    '''
    Returns the smallest signed integer dtype that can store N indexes.
    '''
    if N<=127:
        return np.int8
    elif N<=32727:
        return np.int16
    elif N<=2147483647:
        return np.int32
    else:
        return np.int64