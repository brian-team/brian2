import collections
import weakref

import numpy as np

from brian2.core.base import BrianObject
from brian2.core.namespace import create_namespace
from brian2.core.preferences import brian_prefs
from brian2.core.specifiers import (ArrayVariable, Index, DynamicArrayVariable, 
                                    AttributeValue, Subexpression,
                                    StochasticVariable, SynapticArrayVariable)
from brian2.codegen.languages import PythonLanguage
from brian2.equations.equations import (Equations, DIFFERENTIAL_EQUATION,
                                        STATIC_EQUATION, PARAMETER)
from brian2.groups.group import Group, GroupCodeRunner
from brian2.memory.dynamicarray import DynamicArray1D
from brian2.stateupdaters.base import StateUpdateMethod
from brian2.units.fundamentalunits import Unit
from brian2.units.allunits import second
from brian2.utils.logger import get_logger

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
                                       check_units=False,
                                       template_specifiers=['_num_neurons'])

        self.method = StateUpdateMethod.determine_stateupdater(self.group.equations,
                                                               self.group.namespace,
                                                               self.group.specifiers,
                                                               method)
    
    def update_abstract_code(self):        
        
        self.method = StateUpdateMethod.determine_stateupdater(self.group.equations,
                                                               self.group.namespace,
                                                               self.group.specifiers,
                                                               self.method_choice)
        
        self.abstract_code = self.method(self.group.equations,
                                         self.group.namespace,
                                         self.group.specifiers)


class TargetUpdater(GroupCodeRunner):
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
        self.delays = DynamicArray1D(len(synapses.indices), dtype=np.int16)
        self.queue = SpikeQueue()
        self.spiking_synapses = []
        self.specifiers = {'_spiking_synapses': AttributeValue('_spiking_synapses',
                                                               Unit(1), np.int,
                                                               self, 'spiking_synapses')}
        if objname is None:
            objname = prepost + '*'

        GroupCodeRunner.__init__(self, synapses,
                                 synapses.language.template_synapses,
                                 code=code,
                                 indices=indices,
                                 when=(synapses.clock, 'synapses'),
                                 name=synapses.name + '_' + objname,
                                 template_specifiers=['_num_neurons',
                                                      '_presynaptic',
                                                      '_postsynaptic',
                                                      '_spiking_synapses'])

        # Re-extract the last part of the name from the full name
        self.objname = self.name[len(synapses.name) + 1:]

    def pre_run(self, namespace):
        GroupCodeRunner.pre_run(self, namespace)
        self.queue.compress(self.delays, self.synapse_indices)
    
    def pre_update(self):
        # Push new spikes into the queue
        spikes = self.source.spikes
        if len(spikes):
            indices = np.hstack((self.synapse_indices[spike]
                                 for spike in spikes))
            if len(indices):
                delays = self.delays[indices]
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


class SynapticIndices(object):
    '''
    Convenience object to store the synaptic indices.

    Parameters
    ----------
    source_len : int
        The number of neurons in the presynaptic group.
    target_len : int
        The number of neurons in the postsyanptic group.
    '''
    def __init__(self, source_len, target_len):
        self.source_len = source_len
        self.target_len = target_len
        dtype = smallest_inttype(MAX_SYNAPSES)
        self.synaptic_pre = DynamicArray1D(0, dtype=dtype)
        self.synaptic_post = DynamicArray1D(0, dtype=dtype)
        self.pre_synaptic = [DynamicArray1D(0, dtype=dtype)
                             for _ in xrange(source_len)]
        self.post_synaptic = [DynamicArray1D(0, dtype=dtype)
                              for _ in xrange(target_len)]
        self.i = IndexView(self, self.synaptic_pre)
        self.j = IndexView(self, self.synaptic_post)

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

    def _add_synapses(self, sources, targets):
        new_synapses = len(sources)
        assert new_synapses == len(targets)

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

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        '''
        Returns synaptic indices for `index`, which can be a tuple of indices
        (including arrays and slices), a single index or a string.

        '''
        if isinstance(index, tuple):
            if len(index) == 2:  # two indices (pre- and postsynaptic cell)
                index = (index[0], index[1], slice(None))
            elif len(index) > 3:
                raise IndexError('Need 1, 2 or 3 indices, got %d.' % len(index))

            # Interpret indices
            raise NotImplementedError()
        elif isinstance(index, basestring):
            # interpret the string expression
            raise NotImplementedError()
        elif isinstance(index, (int, np.ndarray, slice, collections.Sequence)):
            return index
        else:
            raise IndexError('Unsupported index type {}'.format(type(index)))


class Synapses(BrianObject, Group):

    def __init__(self, source, target=None, equations=None, pre=None, post=None,
                 namespace=None, dtype=None, language=None,
                 max_delay=0*second, clock=None, method=None, name='synapses*'):
        
        BrianObject.__init__(self, when=clock, name=name)

        if not hasattr(source, 'spikes') and hasattr(source, 'clock'):
            raise TypeError(('Source has to be a SpikeSource with spikes and'
                             ' clock attribute. Is type %r instead')
                            % type(source))

        self.source = weakref.proxy(source)
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
        
        self.equations = equations

        ##### Setup the memory
        self.arrays = self._allocate_memory(dtype=dtype)

        # Setup the namespace
        self.namespace = create_namespace(1, namespace)  #FIXME

        # Code generation (TODO: this should be refactored and modularised)
        # Temporary, set default language to Python
        if language is None:
            language = PythonLanguage()
        self.language = language
        
        self._queues = {}
        self._delays = {}

        self.indices = SynapticIndices(len(source), len(target))

        # Setup specifiers
        self.specifiers = self._create_specifiers()

        #: List of names of all updaters, e.g. ['pre', 'post']
        self._updaters = []
        if pre:
            self.add_pre(pre)
        if post:            
            self.add_post(post)

        #: Performs numerical integration step
        self.state_updater = StateUpdater(self, method)        
        self.contained_objects.append(self.state_updater)
        
        # Activate name attribute access
        Group.__init__(self)

    N = property(fget=lambda self: self.indices.N,
                 doc='Total number of synapses')

    def __len__(self):
        return self.N

    def add_pre(self, code, objname=None):
        '''
        Add code for presynaptic spikes.

        Parameters
        ----------
        code : str
            The abstract code that should be executed on the arrival of
            presynaptic spikes.
        objname : str, optional
            A name for the object, see `TargetUpdater` for more details.

        Returns
        -------
        objname : str
            The final name for the object. Equals `objname` if it was explicitly
            given (and did not end in a wildcard character).

        '''
        return self._add_updater(code, 'pre', objname)

    def add_post(self, code, objname=None):
        '''
        Add code for postsynaptic spikes.

        Parameters
        ----------
        code : str
            The abstract code that should be executed on the arrival of
            postsynaptic spikes.
        objname : str, optional
            A name for the object, see `TargetUpdater` for more details.

        Returns
        -------
        objname : str
            The final name for the object. Equals `objname` if it was explicitly
            given (and did not end in a wildcard character).
        '''
        return self._add_updater(code, 'post', objname)

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
            A name for the object, see `TargetUpdater` for more details.

        Returns
        -------
        objname : str
            The final name for the object. Equals `objname` if it was explicitly
            given (and did not end in a wildcard character).

        '''
        updater = TargetUpdater(self, code, prepost, objname)
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
        for name, spec in self.source.specifiers.iteritems():
            if isinstance(spec, ArrayVariable):
                new_spec = ArrayVariable(spec.name, spec.unit, spec.dtype,
                                         spec.array, '_presynaptic_idx')
                s[name + '_pre'] = new_spec
        for name, spec in self.target.specifiers.iteritems():
            if isinstance(spec, ArrayVariable):
                new_spec = ArrayVariable(spec.name, spec.unit, spec.dtype,
                             spec.array, '_postsynaptic_idx')
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
                  '_presynaptic': DynamicArrayVariable('_presynaptic', Unit(1),
                                                       np.int32,
                                                       self.indices.synaptic_pre,
                                                       '_presynaptic_idx'),
                  '_postsynaptic': DynamicArrayVariable('_postsynaptic', Unit(1),
                                                        np.int32,
                                                        self.indices.synaptic_post,
                                                        '_postsynaptic_idx')})

        for eq in self.equations.itervalues():
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

        self.indices._add_synapses(np.arange(len(self.source)),
                                   np.arange(len(self.target)))

        new_synapses = len(self.source)

        for array in self.arrays.itervalues():
            array.resize(new_synapses)

        for updater in self._updaters:
            getattr(self, updater).delays.resize(new_synapses)


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