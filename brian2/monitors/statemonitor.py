import weakref
import collections

import numpy as np

from brian2.core.variables import Variable, AttributeVariable, ArrayVariable
from brian2.core.base import BrianObject
from brian2.core.scheduler import Scheduler
from brian2.core.preferences import brian_prefs
from brian2.units.fundamentalunits import Unit, Quantity, have_same_dimensions
from brian2.units.allunits import second
from brian2.memory.dynamicarray import DynamicArray, DynamicArray1D
from brian2.groups.group import create_runner_codeobj

__all__ = ['StateMonitor']


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

        if item == 't':
            return Quantity(self.monitor._t.data.copy(), dim=second.dim)
        elif item == 't_':
            return self.monitor._t.data.copy()
        elif item in self.monitor.record_variables:
            unit = self.monitor.variables[item].unit
            return Quantity(self.monitor._values[item].data.T[self.indices].copy(),
                            dim=unit.dim)
        elif item.endswith('_') and item[:-1] in self.monitor.record_variables:
            return self.monitor._values[item[:-1]].data.T[self.indices].copy()
        else:
            raise AttributeError('Unknown attribute %s' % item)

    def _calc_indices(self, item):
        '''
        Convert the neuron indices to indices into the stored values. For example, if neurons [0, 5, 10] have been
        recorded, [5, 10] is converted to [1, 2].
        '''
        if isinstance(item, int):
            indices = np.nonzero(self.monitor.indices == item)[0]
            if len(indices) == 0:
                raise IndexError('Neuron number %d has not been recorded' % item)
            return indices[0]

        if self.monitor.record_all:
            return item
        indices = []
        for index in item:
            if index in self.monitor.indices:
                indices.append(np.nonzero(self.monitor.indices == index)[0][0])
            else:
                raise IndexError('Neuron number %d has not been recorded' % index)
        return np.array(indices)

    def __repr__(self):
        description = '<{classname}, giving access to elements {elements} recorded by {monitor}>'
        return description.format(classname=self.__class__.__name__,
                                  elements=repr(self.item),
                                  monitor=self.monitor.name)


class StateMonitor(BrianObject):
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

        BrianObject.__init__(self, when=scheduler, name=name)

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
            record = np.array(record, dtype=np.int32)
            
        #: The array of recorded indices
        self.indices = record
        # create data structures
        self.reinit()
        
        # Setup variables
        self.variables = {}
        for varname in variables:
            var = source.variables[varname]
            if not (np.issubdtype(var.dtype, np.float64) and
                        np.issubdtype(np.float64, var.dtype)):
                raise NotImplementedError(('Cannot record %s with data type '
                                           '%s, currently only values stored as '
                                           'doubles can be recorded.') %
                                          (varname, var.dtype))
            self.variables[varname] = var
            self.variables['_recorded_'+varname] = Variable(Unit(1),
                                                            self._values[varname])

        self.variables['_t'] = Variable(Unit(1), self._t)
        self.variables['_clock_t'] = AttributeVariable(second, self.clock, 't_')
        self.variables['_indices'] = ArrayVariable('_indices', Unit(1),
                                                   self.indices,
                                                   group=None,
                                                   constant=True)

        self._group_attribute_access_active = True

    def reinit(self):
        self._values = dict((v, DynamicArray((0, len(self.indices)),
                                             use_numpy_resize=True,
                                             dtype=self.source.variables[v].dtype))
                            for v in self.record_variables)
        self._t = DynamicArray1D(0, use_numpy_resize=True,
                                 dtype=brian_prefs['core.default_scalar_dtype'])
    
    def pre_run(self, namespace):
        # Some dummy code so that code generation takes care of the indexing
        # and subexpressions
        code = ['_to_record_%s = %s' % (v, v)
                for v in self.record_variables]
        code += ['_recorded_%s = _recorded_%s' % (v, v)
                 for v in self.record_variables]
        code = '\n'.join(code)
        self.codeobj = create_runner_codeobj(self.source,
                                             code,
                                             name=self.name,
                                             additional_variables=self.variables,
                                             additional_namespace=namespace,
                                             template_name='statemonitor',
                                             indices=self.source.indices,
                                             variable_indices=self.source.variable_indices,
                                             iterate_all=[],
                                             template_kwds={'_variable_names':
                                                                self.record_variables},
                                             #check_units=False,
                                             codeobj_class=self.codeobj_class)

    def update(self):
        self.codeobj()

    def __getitem__(self, item):
        if isinstance(item, (int, np.ndarray)):
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

        # TODO: Decide about the interface
        if item == 't':
            return Quantity(self._t.data.copy(), dim=second.dim)
        elif item == 't_':
            return self._t.data.copy()
        elif item in self.record_variables:
            unit = self.variables[item].unit
            if have_same_dimensions(unit, 1):
                return self._values[item].data.T.copy()
            else:
                return Quantity(self._values[item].data.T.copy(),
                                dim=unit.dim)
        elif item.endswith('_') and item[:-1] in self.record_variables:
            return self._values[item[:-1]].data.T.copy()
        else:
            raise AttributeError('Unknown attribute %s' % item)

    def __repr__(self):
        description = '<{classname}, recording {variables} from {source}>'
        return description.format(classname=self.__class__.__name__,
                                  variables=repr(self.record_variables),
                                  source=self.source.name)
