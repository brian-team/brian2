import collections
import numbers

import numpy as np

from brian2.core.variables import (Variables, Subexpression, get_dtype)
from brian2.groups.group import Group, CodeRunner
from brian2.utils.logger import get_logger
from brian2.units.fundamentalunits import Unit, Quantity
from brian2.units.allunits import second

__all__ = ['StateMonitor']

logger = get_logger(__name__)


class StateMonitorView(object):
    def __init__(self, monitor, item):
        self.monitor = monitor
        self.item = item
        self.indices = self._calc_indices(item)
        self._group_attribute_access_active = True

    def __getattr__(self, item):
        # We do this because __setattr__ and __getattr__ are not active until
        # _group_attribute_access_active attribute is set, and if it is set,
        # then __getattr__ will not be called. Therefore, if getattr is called
        # with this name, it is because it hasn't been set yet and so this
        # method should raise an AttributeError to agree that it hasn't been
        # called yet.
        if item == '_group_attribute_access_active':
            raise AttributeError
        if not hasattr(self, '_group_attribute_access_active'):
            raise AttributeError

        mon = self.monitor
        if item == 't':
            return Quantity(mon.variables['t'].get_value(), dim=second.dim)
        elif item == 't_':
            return mon.variables['t'].get_value()
        elif item in mon.record_variables:
            dims = mon.variables[item].dim
            return Quantity(mon.variables[item].get_value().T[self.indices],
                            dim=dims, copy=True)
        elif item.endswith('_') and item[:-1] in mon.record_variables:
            return mon.variables[item[:-1]].get_value().T[self.indices].copy()
        else:
            raise AttributeError('Unknown attribute %s' % item)

    def _calc_indices(self, item):
        '''
        Convert the neuron indices to indices into the stored values. For example, if neurons [0, 5, 10] have been
        recorded, [5, 10] is converted to [1, 2].
        '''
        dtype = get_dtype(item)
        # scalar value
        if np.issubdtype(dtype, np.signedinteger) and not isinstance(item, np.ndarray):
            indices = np.nonzero(self.monitor.record == item)[0]
            if len(indices) == 0:
                raise IndexError('Index number %d has not been recorded' % item)
            return indices[0]

        if self.monitor.record_all:
            return item
        indices = []
        for index in item:
            if index in self.monitor.record:
                indices.append(np.nonzero(self.monitor.record == index)[0][0])
            else:
                raise IndexError('Index number %d has not been recorded' % index)
        return np.array(indices)

    def __repr__(self):
        description = '<{classname}, giving access to elements {elements} recorded by {monitor}>'
        return description.format(classname=self.__class__.__name__,
                                  elements=repr(self.item),
                                  monitor=self.monitor.name)


class StateMonitor(Group, CodeRunner):
    '''
    Record values of state variables during a run
    
    To extract recorded values after a run, use the ``t`` attribute for the
    array of times at which values were recorded, and variable name attribute
    for the values. The values will have shape ``(len(indices), len(t))``,
    where ``indices`` are the array indices which were recorded. When indexing
    the `StateMonitor` directly, the returned object can be used to get the
    recorded values for the specified indices, i.e. the indexing semantic
    refers to the indices in ``source``, not to the relative indices of the
    recorded values. For example, when recording only neurons with even numbers,
    `mon[[0, 2]].v` will return the values for neurons 0 and 2, whereas
    `mon.v[[0, 2]]` will return the values for the first and third *recorded*
    neurons, i.e. for neurons 0 and 4.

    Parameters
    ----------
    source : `Group`
        Which object to record values from.
    variables : str, sequence of str, True
        Which variables to record, or ``True`` to record all variables
        (note that this may use a great deal of memory).
    record : bool, sequence of ints
        Which indices to record, nothing is recorded for ``False``,
        everything is recorded for ``True`` (warning: may use a great deal of
        memory), or a specified subset of indices.
    dt : `Quantity`, optional
        The time step to be used for the monitor. Cannot be combined with
        the `clock` argument.
    clock : `Clock`, optional
        The update clock to be used. If neither a clock, nor the ``dt`` argument
        is specified, the clock of the `source` will be used.
    when : str, optional
        At which point during a time step the values should be recorded.
        Defaults to ``'start'``.
    order : int, optional
        The priority of of this group for operations occurring at the same time
        step and in the same scheduling slot. Defaults to 0.
    name : str, optional
        A unique name for the object, otherwise will use
        ``source.name+'statemonitor_0'``, etc.
    codeobj_class : `CodeObject`, optional
        The `CodeObject` class to create.

    Examples
    --------
    
    Record all variables, first 5 indices::
    
        eqs = """
        dV/dt = (2-V)/(10*ms) : 1
        """
        threshold = 'V>1'
        reset = 'V = 0'
        G = NeuronGroup(100, eqs, threshold=threshold, reset=reset)
        G.V = rand(len(G))
        M = StateMonitor(G, True, record=range(5))
        run(100*ms)
        plot(M.t, M.V.T)
        show()

    Notes
    -----

    Since this monitor by default records in the ``'start'`` time slot,
    recordings of the membrane potential in integrate-and-fire models may look
    unexpected: the recorded membrane potential trace will never be above
    threshold in an integrate-and-fire model, because the reset statement will
    have been applied already. Set the ``when`` keyword to a different value if
    this is not what you want.

    Note that ``record=True`` only works in runtime mode for synaptic variables.
    This is because the actual array of indices has to be calculated and this is
    not possible in standalone mode, where the synapses have not been created
    yet at this stage. Consider using an explicit array of indices instead,
    i.e. something like ``record=np.arange(n_synapses)``.
    '''
    invalidates_magic_network = False
    add_to_magic_network = True
    def __init__(self, source, variables, record, dt=None, clock=None,
                 when='start', order=0, name='statemonitor*', codeobj_class=None):
        self.source = source
        # Make the monitor use the explicitly defined namespace of its source
        # group (if it exists)
        self.namespace = getattr(source, 'namespace', None)
        self.codeobj_class = codeobj_class

        # run by default on source clock at the end
        if dt is None and clock is None:
            clock = source.clock

        # variables should always be a list of strings
        if variables is True:
            variables = source.equations.names
        elif isinstance(variables, str):
            variables = [variables]
        #: The variables to record
        self.record_variables = variables

        # record should always be an array of ints
        self.record_all = False
        if hasattr(record, '_indices'):
            # The ._indices method always returns absolute indices
            # If the source is already a subgroup of another group, we therefore
            # have to shift the indices to become relative to the subgroup
            record = record._indices() - getattr(source, '_offset', 0)
        if record is True:
            self.record_all = True
            try:
                record = np.arange(len(source), dtype=np.int32)
            except NotImplementedError:
                # In standalone mode, this is not possible for synaptic
                # variables because the number of synapses is not defined yet
                raise NotImplementedError(('Cannot determine the actual '
                                           'indices to record for record=True. '
                                           'This can occur for example in '
                                           'standalone mode when trying to '
                                           'record a synaptic variable. '
                                           'Consider providing an explicit '
                                           'array of indices for the record '
                                           'argument.'))
        elif record is False:
            record = np.array([], dtype=np.int32)
        elif isinstance(record, numbers.Number):
            record = np.array([record], dtype=np.int32)
        else:
            record = np.asarray(record, dtype=np.int32)

        #: The array of recorded indices
        self.record = record
        self.n_indices = len(record)

        # Some dummy code so that code generation takes care of the indexing
        # and subexpressions
        code = ['_to_record_%s = _source_%s' % (v, v)
                for v in variables]
        code = '\n'.join(code)

        CodeRunner.__init__(self, group=self, template='statemonitor',
                            code=code, name=name,
                            clock=clock,
                            dt=dt,
                            when=when,
                            order=order,
                            check_units=False)

        self.add_dependency(source)

        # Setup variables
        self.variables = Variables(self)

        self.variables.add_dynamic_array('t', size=0, dimensions=second.dim,
                                         constant=False,
                                         dtype=self._clock.variables['t'].dtype)
        self.variables.add_array('N', dtype=np.int32, size=1, scalar=True,
                                 read_only=True)
        self.variables.add_array('_indices', size=len(self.record),
                                 dtype=self.record.dtype, constant=True,
                                 read_only=True, values=self.record)
        self.variables.create_clock_variables(self._clock,
                                              prefix='_clock_')
        for varname in variables:
            var = source.variables[varname]
            if var.scalar and len(self.record) > 1:
                logger.warn(('Variable %s is a shared variable but it will be '
                             'recorded once for every target.' % varname),
                            once=True)
            index = source.variables.indices[varname]
            self.variables.add_reference('_source_%s' % varname,
                                         source, varname, index=index)
            if not index in ('_idx', '0') and index not in variables:
                self.variables.add_reference(index, source)
            self.variables.add_dynamic_array(varname,
                                             size=(0, len(self.record)),
                                             resize_along_first=True,
                                             dimensions=var.dim,
                                             dtype=var.dtype,
                                             constant=False,
                                             read_only=True)

        for varname in variables:
            var = self.source.variables[varname]
            self.variables.add_auxiliary_variable('_to_record_' + varname,
                                                  dimensions=var.dim,
                                                  dtype=var.dtype,
                                                  scalar=var.scalar)

        self.recorded_variables = dict([(varname, self.variables[varname])
                                        for varname in variables])
        recorded_names = [varname for varname in variables]

        self.needed_variables = recorded_names
        self.template_kwds = {'_recorded_variables': self.recorded_variables}
        self.written_readonly_vars = {self.variables[varname]
                                      for varname in self.record_variables}
        self._enable_group_attributes()

    def resize(self, new_size):
        self.variables['N'].set_value(new_size)
        self.variables['t'].resize(new_size)

        for var in self.recorded_variables.values():
            var.resize((new_size, self.n_indices))

    def reinit(self):
        raise NotImplementedError()

    def __getitem__(self, item):
        dtype = get_dtype(item)
        if np.issubdtype(dtype, np.signedinteger):
            return StateMonitorView(self, item)
        elif isinstance(item, collections.Sequence):
            index_array = np.array(item)
            if not np.issubdtype(index_array.dtype, np.signedinteger):
                raise TypeError('Index has to be an integer or a sequence '
                                'of integers')
            return StateMonitorView(self, item)
        elif hasattr(item, '_indices'):
            # objects that support the indexing interface will return absolute
            # indices but here we need relative ones
            # TODO: How to we prevent the use of completely unrelated objects here?
            source_offset = getattr(self.source, '_offset', 0)
            return StateMonitorView(self, item._indices() - source_offset)
        else:
            raise TypeError('Cannot use object of type %s as an index'
                            % type(item))

    def __getattr__(self, item):
        # We do this because __setattr__ and __getattr__ are not active until
        # _group_attribute_access_active attribute is set, and if it is set,
        # then __getattr__ will not be called. Therefore, if getattr is called
        # with this name, it is because it hasn't been set yet and so this
        # method should raise an AttributeError to agree that it hasn't been
        # called yet.
        if item == '_group_attribute_access_active':
            raise AttributeError
        if not hasattr(self, '_group_attribute_access_active'):
            raise AttributeError
        if item in self.record_variables:
            var_dim = self.variables[item].dim
            return Quantity(self.variables[item].get_value().T,
                            dim=var_dim, copy=True)
        elif item.endswith('_') and item[:-1] in self.record_variables:
            return self.variables[item[:-1]].get_value().T
        else:
            return Group.__getattr__(self, item)

    def __repr__(self):
        description = '<{classname}, recording {variables} from {source}>'
        return description.format(classname=self.__class__.__name__,
                                  variables=repr(self.record_variables),
                                  source=self.source.name)

    def record_single_timestep(self):
        '''
        Records a single time step. Useful for recording the values at the end
        of the simulation -- otherwise a `StateMonitor` will not record the
        last simulated values since its ``when`` attribute defaults to
        ``'start'``, i.e. the last recording is at the *beginning* of the last
        time step.

        Notes
        -----
        This function will only work if the `StateMonitor` has been already run,
        but a run with a length of ``0*ms`` does suffice.

        Examples
        --------
        >>> from brian2 import *
        >>> G = NeuronGroup(1, 'dv/dt = -v/(5*ms) : 1')
        >>> G.v = 1
        >>> mon = StateMonitor(G, 'v', record=True)
        >>> run(0.5*ms)
        >>> print(np.array_str(mon.v[:], precision=3))
        [[ 1.     0.98   0.961  0.942  0.923]]
        >>> print(mon.t[:])
        [   0.  100.  200.  300.  400.] us
        >>> print(np.array_str(G.v[:], precision=3))  # last value had not been recorded
        [ 0.905]
        >>> mon.record_single_timestep()
        >>> print(mon.t[:])
        [   0.  100.  200.  300.  400.  500.] us
        >>> print(np.array_str(mon.v[:], precision=3))
        [[ 1.     0.98   0.961  0.942  0.923  0.905]]
        '''
        if self.codeobj is None:
            raise TypeError('Can only record a single time step after the '
                            'network has been run once.')
        self.codeobj()
