import weakref

import numpy as np

from brian2.codegen.codeobject import create_codeobject
from brian2.core.specifiers import (Value,
                                    ReadOnlyValue, ArrayVariable,
                                    AttributeValue, Index)
from brian2.core.base import BrianObject
from brian2.core.scheduler import Scheduler
from brian2.groups.group import Group
from brian2.units.fundamentalunits import Unit
from brian2.units.allunits import second
from brian2.memory.dynamicarray import DynamicArray, DynamicArray1D

__all__ = ['StateMonitor']


class MonitorVariable(Value):
    def __init__(self, name, unit, dtype, monitor):
        Value.__init__(self, name, unit, dtype)
        self.monitor = weakref.proxy(monitor)
    
    def get_value(self):
        return self.monitor._values[self.name]


class MonitorTime(Value):
    def __init__(self, monitor):
        Value.__init__(self, 't', second, np.float64)
        self.monitor = weakref.proxy(monitor)

    def get_value(self):
        return self.monitor._t[:]

class StateMonitor(BrianObject, Group):
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
            record = np.array([], dtype=int)
        elif record is True:
            self.record_all = True
            record = np.arange(len(source))
        else:
            record = np.array(record, dtype=int)
            
        #: The array of recorded indices
        self.indices = record
        
        # create data structures
        self.reinit()
        
        # initialise Group access
        self.specifiers = {}
        for idx, variable in enumerate(variables):
            spec = source.specifiers[variable]
            self.specifiers['_source_' + variable] = ArrayVariable(variable,
                                                                   spec.unit,
                                                                   spec.dtype,
                                                                   spec.array,
                                                                   '_record_idx',
                                                                   group=spec.group,
                                                                   constant=spec.constant,
                                                                   scalar=spec.scalar,
                                                                   is_bool=spec.is_bool)
            self.specifiers[variable] = MonitorVariable(variable,
                                                        spec.unit,
                                                        spec.dtype,
                                                        self)
            self.specifiers['_recorded_'+variable] = ReadOnlyValue('_recorded_'+variable, Unit(1),
                                                                   self._values[variable].dtype,
                                                                   self._values[variable])
        self.specifiers['_t'] = ReadOnlyValue('_t', Unit(1), self._t.dtype,
                                              self._t)

        self.specifiers['_clock_t'] = AttributeValue('t',  second, np.float64,
                                                     self.clock, 't_')

        self.specifiers['t'] = MonitorTime(self)

        Group.__init__(self)

    def reinit(self):
        self._values = dict((v, DynamicArray((0, len(self.variables))))
                            for v in self.variables)
        self._t = DynamicArray1D(0)
    
    def pre_run(self, namespace):
        # Some dummy code so that code generation takes care of the indexing
        # and subexpressions
        code = ['_to_record_%s = _source_%s' % (v, v)
                for v in self.variables]
        code = '\n'.join(code)
        self.codeobj = create_codeobject(self.name,
                                         code,
                                         {}, # no namespace
                                         self.specifiers,
                                         template_name='statemonitor',
                                         indices={'_record_idx': Index('_record_idx', self.record_all)},
                                         template_kwds={'_variable_names': self.variables},
                                         codeobj_class=self.codeobj_class)

    def update(self):
        self.codeobj()

    def __repr__(self):
        description = '<{classname}, recording {variables} from {source}>'
        return description.format(classname=self.__class__.__name__,
                                  variables=repr(self.variables),
                                  source=self.source.name)
