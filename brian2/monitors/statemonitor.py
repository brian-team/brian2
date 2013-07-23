import weakref

import numpy as np

from brian2.core.specifiers import (ReadOnlyValue, ArrayVariable,
                                    AttributeValue, Index)
from brian2.core.base import BrianObject
from brian2.core.scheduler import Scheduler
from brian2.core.preferences import brian_prefs
from brian2.units.fundamentalunits import Unit, Quantity, have_same_dimensions
from brian2.units.allunits import second
from brian2.memory.dynamicarray import DynamicArray, DynamicArray1D
from brian2.groups.group import create_runner_codeobj

__all__ = ['StateMonitor']


class StateMonitor(BrianObject):
    '''
    Record values of state variables during a run
    
    To extract recorded values after a run, use `t` attribute for the
    array of times at which values were recorded, and variable name attribute
    for the values. The values will have shape ``(len(t), len(indices))``,
    where `indices` are the array indices which were recorded.

    Parameters
    ----------
    source : `NeuronGroup`, `Group`
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
        plot(M.t, M.V)
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
        self.variables = variables

        # record should always be an array of ints
        self.record_all = False
        if record is None or record is False:
            record = np.array([], dtype=np.int32)
        elif record is True:
            self.record_all = True
            record = np.arange(len(source), dtype=np.int32)
        else:
            record = np.array(record, dtype=np.int32)
            
        #: The array of recorded indices
        self.indices = record
        
        # create data structures
        self.reinit()
        
        # Setup specifiers
        self.specifiers = {}
        for variable in variables:
            spec = source.specifiers[variable]
            self.specifiers[variable] = weakref.proxy(spec)

            self.specifiers['_recorded_'+variable] = ReadOnlyValue('_recorded_'+variable, Unit(1),
                                                                   self._values[variable].dtype,
                                                                   self._values[variable])

        self.specifiers['_t'] = ReadOnlyValue('_t', Unit(1), self._t.dtype,
                                              self._t)
        self.specifiers['_clock_t'] = AttributeValue('t',  second, np.float64,
                                                     self.clock, 't_')
        self.specifiers['_indices'] = ArrayVariable('_indices', Unit(1),
                                                    np.int32, self.indices,
                                                    index='', group=None,
                                                    constant=True)

        self._group_attribute_access_active = True

    def reinit(self):
        self._values = dict((v, DynamicArray((0, len(self.indices)),
                                             use_numpy_resize=True,
                                             dtype=self.source.specifiers[v].dtype))
                            for v in self.variables)
        self._t = DynamicArray1D(0, use_numpy_resize=True,
                                 dtype=brian_prefs['core.default_scalar_dtype'])
    
    def pre_run(self, namespace):
        # Some dummy code so that code generation takes care of the indexing
        # and subexpressions
        code = ['_to_record_%s = %s' % (v, v)
                for v in self.variables]
        code += ['_recorded_%s = _recorded_%s' % (v, v)
                 for v in self.variables]
        code = '\n'.join(code)
        self.codeobj = create_runner_codeobj(self.source,
                                         code,
                                         name=self.name,
                                         additional_specifiers=self.specifiers,
                                         additional_namespace=namespace,
                                         template_name='statemonitor',
                                         indices={'_neuron_idx': Index('_neuron_idx', self.record_all)},
                                         template_kwds={'_variable_names': self.variables},
                                         codeobj_class=self.codeobj_class)

    def update(self):
        self.codeobj()

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
        elif item in self.variables:
            unit = self.specifiers[item].unit
            if have_same_dimensions(unit, 1):
                return self._values[item].data.copy()
            else:
                return Quantity(self._values[item].data.copy(),
                                dim=unit.dim)
        elif item.endswith('_') and item[:-1] in self.variables:
            return self._values[item[:-1]].data.copy()
        else:
            raise AttributeError('Unknown attribute %s' % item)

    def __repr__(self):
        description = '<{classname}, recording {variables} from {source}>'
        return description.format(classname=self.__class__.__name__,
                                  variables=repr(self.variables),
                                  source=self.source.name)
