import weakref

from numpy import array, arange

from brian2.core.specifiers import Value
from brian2.core.base import BrianObject
from brian2.core.scheduler import Scheduler
from brian2.groups.group import Group
from brian2.units.allunits import second

__all__ = ['StateMonitor']

class MonitorVariable(Value):
    def __init__(self, name, unit, dtype, monitor):
        Value.__init__(self, name, unit, dtype)
        self.monitor = weakref.proxy(monitor)
    
    def get_value(self):
        return array(self.monitor._values[self.name])


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
        
    Notes
    -----

    TODO: multiple features, below:
    
    * Cacheing extracted values (t, V, etc.)
    * Improve efficiency by using dynamic arrays instead of lists?
    '''
    def __init__(self, source, variables, record=None, when=None,
                 name='statemonitor*'):
        self.source = weakref.proxy(source)

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
        if record is None or record is False:
            record = array([], dtype=int)
        elif record is True:
            record = arange(len(source))
        else:
            record = array(record, dtype=int)
            
        #: The array of recorded indices
        self.indices = record
        
        # create data structures
        self.reinit()
        
        # initialise Group access
        self.specifiers = {}
        for variable in variables:
            spec = source.specifiers[variable]
            self.specifiers[variable] = MonitorVariable(variable,
                                                        spec.unit,
                                                        spec.dtype,
                                                        self)        
        Group.__init__(self)
        
    def reinit(self):
        self._values = dict((var, []) for var in self.variables)
        self._t = []
    
    def update(self):
        for var in self.variables:
            self._values[var].append(self.source.state_(var)[self.indices])
        self._t.append(self.clock.t_)
        
    @property
    def t(self):
        '''
        Array of record times.
        '''
        return array(self._t)*second
    
    @property
    def t_(self):
        '''
        Array of record times (without units).
        '''
        return array(self._t)

    def __repr__(self):
        description = '<{classname}, recording {variables} from {source}>'
        return description.format(classname=self.__class__.__name__,
                                  variables=repr(self.variables),
                                  source=self.source.name)


if __name__=='__main__':
    from pylab import *
    from brian2 import *
    from brian2.codegen.languages import *
    import time

    N = 100
    tau = 10*ms
    eqs = '''
    dV/dt = (2*volt-V)/tau : volt
    '''
    threshold = 'V>1'
    reset = 'V = 0'
    G = NeuronGroup(N, eqs, threshold=threshold, reset=reset)
    G.V = rand(N)*volt
    M = StateMonitor(G, True, record=range(5))
    run(100*ms)
    print M.V.shape
    plot(M.t, M.V)
    show()
