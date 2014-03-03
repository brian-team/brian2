import weakref
import collections

import numpy as np

from brian2.core.variables import (Variables, Subexpression, get_dtype)
from brian2.core.scheduler import Scheduler
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
            return mon.t
        elif item == 't_':
            return mon.t_
        elif item in mon.record_variables:
            unit = mon.variables[item].unit
            return Quantity(mon.variables['_recorded_'+item].get_value().T[self.indices],
                            dim=unit.dim, copy=True)
        elif item.endswith('_') and item[:-1] in mon.record_variables:
            return mon.variables['_recorded_'+item[:-1]].get_value().T[self.indices].copy()
        else:
            raise AttributeError('Unknown attribute %s' % item)

    def _calc_indices(self, item):
        '''
        Convert the neuron indices to indices into the stored values. For example, if neurons [0, 5, 10] have been
        recorded, [5, 10] is converted to [1, 2].
        '''
        dtype = get_dtype(item)
        # scalar value
        if np.issubdtype(dtype, np.int) and not isinstance(item, np.ndarray):
            indices = np.nonzero(self.monitor.indices == item)[0]
            if len(indices) == 0:
                raise IndexError('Index number %d has not been recorded' % item)
            return indices[0]

        if self.monitor.record_all:
            return item
        indices = []
        for index in item:
            if index in self.monitor.indices:
                indices.append(np.nonzero(self.monitor.indices == index)[0][0])
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
    
    To extract recorded values after a run, use `t` attribute for the
    array of times at which values were recorded, and variable name attribute
    for the values. The values will have shape ``(len(indices), len(t))``,
    where `indices` are the array indices which were recorded. When indexing the
    `StateMonitor` directly, the returned object can be used to get the
    recorded values for the specified indices, i.e. the indexing semantic
    refers to the indices in `source`, not to the relative indices of the
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
    record : None, False, True, sequence of ints
        Which indices to record, nothing is recorded for ``None`` or ``False``,
        everything is recorded for ``True`` (warning: may use a great deal of
        memory), or a specified subset of indices.
    when : `Scheduler`, optional
        When to record the spikes, by default uses the clock of the source
        and records spikes in the slot 'end'.
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

    '''
    def __init__(self, source, variables, record=None, when=None,
                 name='statemonitor*', codeobj_class=None):
        self.source = weakref.proxy(source)
        self.codeobj_class = codeobj_class

        # run by default on source clock at the end
        scheduler = Scheduler(when)
        if not scheduler.defined_clock:
            scheduler.clock = source.clock
        if not scheduler.defined_when:
            scheduler.when = 'end'

        # variables should always be a list of strings
        if variables is True:
            variables = source.equations.names
        elif isinstance(variables, str):
            variables = [variables]
        #: The variables to record
        self.record_variables = variables

        # record should always be an array of ints
        self.record_all = False
        if record is True:
            self.record_all = True
            record = np.arange(len(source), dtype=np.int32)
        elif record is None or record is False:
            record = np.array([], dtype=np.int32)
        elif isinstance(record, int):
            record = np.array([record], dtype=np.int32)
        else:
            record = np.asarray(record, dtype=np.int32)
            
        #: The array of recorded indices
        self.indices = record
        self.n_indices = len(record)

        # Some dummy code so that code generation takes care of the indexing
        # and subexpressions
        code = ['_to_record_%s = %s' % (v, v)
                for v in self.record_variables]
        code = '\n'.join(code)

        CodeRunner.__init__(self, group=self, template='statemonitor',
                            code=code, name=name,
                            when=scheduler,
                            check_units=False)

        # Setup variables
        self.variables = Variables(self)

        self.variables.add_dynamic_array('t', size=0, unit=second,
                                         constant=False, constant_size=False)
        if scheduler.clock is source.clock:
            self.variables.add_reference('_clock_t', source.variables['t'])
        else:
            self.variables.add_attribute_variable('_clock_t', unit=second,
                                                  obj=scheduler.clock,
                                                  attribute='t_')
        self.variables.add_attribute_variable('N', unit=Unit(1),
                                              dtype=np.int32,
                                              obj=self, attribute='_N')
        self.variables.add_array('_indices', size=len(self.indices),
                                 unit=Unit(1), dtype=self.indices.dtype,
                                 constant=True, read_only=True)
        self.variables['_indices'].set_value(self.indices)

        for varname in variables:
            var = source.variables[varname]
            if var.scalar and len(self.indices) > 1:
                logger.warn(('Variable %s is a scalar variable but it will be '
                             'recorded once for every target.' % varname),
                            once=True)
            index = source.variables.indices[varname]
            self.variables.add_reference(varname, var,
                                         index=index)
            if not index in ('_idx', '0') and index not in variables:
                self.variables.add_reference(index, source.variables[index])
            # For subexpressions, we also need all referred variables (if they
            # are not already present, e.g. the t as _clock_t
            if isinstance(var, Subexpression):
                for subexpr_varname in var.identifiers:
                    if subexpr_varname in source.variables:
                        source_var = source.variables[subexpr_varname]
                        index = source.variables.indices[subexpr_varname]
                        if index != '_idx' and index not in variables:
                            self.variables.add_reference(index,
                                                         source.variables[index])
                        if not source_var in self.variables.values():
                            source_index = source.variables.indices[subexpr_varname]
                            # `translate_subexpression` will later replace
                            # the name used in the original subexpression with
                            # _source_varname
                            self.variables.add_reference('_source_'+subexpr_varname,
                                                         source_var,
                                                         index=source_index)
            self.variables.add_dynamic_array('_recorded_' + varname,
                                             size=(0, len(self.indices)),
                                             unit=var.unit,
                                             dtype=var.dtype,
                                             constant=False,
                                             constant_size=False,
                                             is_bool=var.is_bool)

        for varname in self.record_variables:
            var = self.source.variables[varname]
            self.variables.add_auxiliary_variable('_to_record_' + varname,
                                                  unit=var.unit,
                                                  dtype=var.dtype,
                                                  scalar=var.scalar,
                                                  is_bool=var.is_bool)

        self.recorded_variables = dict([(varname,
                                         self.variables['_recorded_'+varname])
                                        for varname in self.record_variables])
        recorded_names = ['_recorded_'+varname
                          for varname in self.record_variables]

        self.needed_variables = recorded_names
        self.template_kwds = template_kwds={'_recorded_variables':
                                            self.recorded_variables}
        self._N = 0
        self._enable_group_attributes()

    def __len__(self):
        return self._N

    def resize(self, new_size):
        self.variables['t'].resize(new_size)

        for var in self.recorded_variables.values():
            var.resize((new_size, self.n_indices))
        self._N = new_size

    def reinit(self):
        raise NotImplementedError()

    def __getitem__(self, item):
        dtype = get_dtype(item)
        if np.issubdtype(dtype, np.int):
            return StateMonitorView(self, item)
        elif isinstance(item, collections.Sequence):
            index_array = np.array(item)
            if not np.issubdtype(index_array.dtype, np.int):
                raise TypeError('Index has to be an integer or a sequence '
                                'of integers')
            return StateMonitorView(self, item)
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
            unit = self.variables[item].unit
            return Quantity(self.variables['_recorded_'+item].get_value().T,
                            dim=unit.dim, copy=True)
        elif item.endswith('_') and item[:-1] in self.record_variables:
            return self.variables['_recorded_'+item[:-1]].get_value().T
        else:
            return Group.__getattr__(self, item)

    def __repr__(self):
        description = '<{classname}, recording {variables} from {source}>'
        return description.format(classname=self.__class__.__name__,
                                  variables=repr(self.record_variables),
                                  source=self.source.name)
