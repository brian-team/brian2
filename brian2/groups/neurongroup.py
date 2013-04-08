'''
This model defines the `NeuronGroup`, the core of most simulations.
'''
import weakref

import numpy as np
from numpy import array

from brian2.equations.equations import (Equations, DIFFERENTIAL_EQUATION,
                                        STATIC_EQUATION, PARAMETER)
from brian2.equations.refractory import add_refractoriness
from brian2.stateupdaters.base import StateUpdateMethod
from brian2.codegen.languages import PythonLanguage
from brian2.memory import allocate_array
from brian2.core.preferences import brian_prefs
from brian2.core.base import BrianObject
from brian2.core.namespace import ObjectWithNamespace
from brian2.core.specifiers import (ReadOnlyValue, AttributeValue, ArrayVariable,
                                    StochasticVariable, Subexpression, Index)
from brian2.core.spikesource import SpikeSource
from brian2.core.scheduler import Scheduler
from brian2.utils.logger import get_logger
from brian2.groups.group import Group
from brian2.units.allunits import second
from brian2.units.fundamentalunits import Unit
from brian2.utils.stringtools import get_identifiers
from brian2.stateupdaters.exact import linear

__all__ = ['NeuronGroup',
           'CodeRunner']

logger = get_logger(__name__)

class CodeRunner(BrianObject):
    '''
    Runs a code object on an update schedule.
    
    Parameters
    ----------
    codeobj : `CodeObject`
        The code to run every timestep.
    when : `Scheduler`
        When to execute the update, see `Scheduler` for details.
    name : str
        The name of this code runner.
    '''
    basename = 'code_runner'
    def __init__(self, codeobj, when=None, name=None):
        BrianObject.__init__(self, when=when, name=name)
        self.codeobj = codeobj

    def pre_update(self):
        pass

    def update(self, **kwds):
        self.pre_update()
        return_value = self.codeobj(**kwds)
        self.post_update(return_value)

    def post_update(self, return_value):
        pass


class NeuronGroupCodeRunner(CodeRunner):
    '''
    A `CodeRunner` for use in `NeuronGroup`. Keeps a reference to the
    `NeuronGroup`.
    '''
    def __init__(self, group, codeobj, when=None, name=None):
        CodeRunner.__init__(self, codeobj, when=when, name=name)
        self.group = weakref.proxy(group)


class StateUpdater(NeuronGroupCodeRunner):
    def pre_update(self):
        self.group.is_active_[:] = self.group.clock.t_ >= self.group.refractory_until_


class Thresholder(NeuronGroupCodeRunner):
    def post_update(self, return_value):
        spikes = return_value
        # Save the spikes in the NeuronGroup so others can use it
        self.group.spikes = spikes
        self.group.refractory_until_[spikes] = self.group.clock.t_ + self.group.refractory_[spikes]


class Resetter(NeuronGroupCodeRunner):
    pass


class NeuronGroup(ObjectWithNamespace, BrianObject, Group, SpikeSource):
    '''
    Group of neurons
    
    In addition to the variable names you create, `NeuronGroup` will have an
    additional state variable ``refractory`` (in units of seconds) which 
    gives the absolute refractory period of the neuron. This value can be
    modified in the reset code. (TODO: more modifiability)
    
    Parameters
    ----------
    N : int
        Number of neurons in the group.
    equations : (str, `Equations`)
        The differential equations defining the group
    method : (str, function), optional
        The numerical integration method. Either a string with the name of a
        registered method (e.g. "euler") or a function that receives an
        `Equations` object and returns the corresponding abstract code. If no
        method is specified, a suitable method will be chosen automatically.
    threshold : str, optional
        The condition which produces spikes. Should be a single line boolean
        expression.
    reset : str, optional
        The (possibly multi-line) string with the code to execute on reset.
    namespace: dict, optional
        A dictionary mapping variable/function names to the respective objects.
        If no `namespace` is given, the "implicit" namespace, consisting of
        the local and global namespace surrounding the creation of the class,
        is used.
    dtype : (`dtype`, `dict`), optional
        The `numpy.dtype` that will be used to store the values, or
        `core.default_scalar_dtype` if not specified (`numpy.float64` by
        default).
    clock : Clock, optional
        The update clock to be used, or defaultclock if not specified.
    name : str, optional
        A unique name for the group, otherwise use ``neurongroup_0``, etc.
        
    Notes
    -----
    `NeuronGroup` contains a `StateUpdater`, `Thresholder` and `Resetter`, and
    these are run at the 'groups', 'thresholds' and 'resets' slots (i.e. the
    values of `Scheduler.when` take these values). The `Scheduler.order`
    attribute is set to 0 initially, but this can be modified using the
    attributes `state_updater`, `thresholder` and `resetter`.    
    '''
    basename = 'neurongroup'
    def __init__(self, N, equations, method=None,
                 threshold=None,
                 reset=None,
                 namespace=None,
                 dtype=None, language=None,
                 clock=None, name=None):
        BrianObject.__init__(self, when=clock, name=name)

        try:
            self.N = N = int(N)
        except ValueError:
            if isinstance(N, str):
                raise TypeError("First NeuronGroup argument should be size, not equations.")
            raise
        if N < 1:
            raise ValueError("NeuronGroup size should be at least 1, was " + str(N))

        ##### Prepare and validate equations
        if isinstance(equations, basestring):
            equations = Equations(equations)
        if not isinstance(equations, Equations):
            raise ValueError(('equations has to be a string or an Equations '
                              'object, is "%s" instead.') % type(equations))
        # add refractoriness
        equations = add_refractoriness(equations)
        self.equations = equations

        logger.debug("Creating NeuronGroup of size {self.N}, "
                     "equations {self.equations}.".format(self=self))

        # Check flags
        equations.check_flags({DIFFERENTIAL_EQUATION: ('active'),
                               PARAMETER: ('constant')})

        ##### Setup the memory
        self.arrays = self.allocate_memory(dtype=dtype)

        #: The array of spikes from the most recent threshold operation
        self.spikes = array([], dtype=int)

        # Setup the namespace
        self.namespace = self.create_namespace(self.N, namespace)

        # Setup specifiers
        self.specifiers = self.create_specifiers()

        # Check units
        self.equations.check_units(self.namespace, self.specifiers)

        # Code generation (TODO: this should be refactored and modularised)
        # Temporary, set default language to Python
        if language is None:
            language = PythonLanguage()
        self.language = language

        #: Performs thresholding step, sets the value of `spikes`
        self.thresholder = self.create_thresholder(threshold)
        #: Resets neurons which have spiked (`spikes`)
        self.resetter = self.create_resetter(reset)

        #: The state update method
        self.method = StateUpdateMethod.determine_stateupdater(self.equations,
                                                               self.namespace,
                                                               self.specifiers,
                                                               method)

        #: Abstract code for one integration step. Will be set in
        #: `create_state_updater`
        self.abstract_code = None

        #: Performs numerical integration step
        self.state_updater = self.create_state_updater()

        # Creation of contained_objects that do the work
        self.contained_objects.append(self.state_updater)
        if self.thresholder is not None:
            self.contained_objects.append(self.thresholder)
        if self.resetter is not None:
            self.contained_objects.append(self.resetter)

        # Activate name attribute access
        Group.__init__(self)


    def __len__(self):
        '''
        Return number of neurons in the group.
        '''
        return self.N


    def allocate_memory(self, dtype=None):
        # Allocate memory (TODO: this should be refactored somewhere at some point)
        arrayvarnames = set(eq.varname for eq in self.equations.itervalues() if
                            eq.eq_type in (DIFFERENTIAL_EQUATION,
                                           PARAMETER))
        arrays = {}
        for name in arrayvarnames:
            if isinstance(dtype, dict):
                curdtype = dtype[name]
            else:
                curdtype = dtype
            if curdtype is None:
                curdtype = brian_prefs['core.default_scalar_dtype']
            arrays[name] = allocate_array(self.N, dtype=curdtype)
        logger.debug("NeuronGroup memory allocated successfully.")
        return arrays


    def create_codeobj(self, name, code, identifiers,
                       template=None, iterate_all=True):
        ''' A little helper function to reduce the amount of repetition when
        calling the language's create_codeobj (always pass self.specifiers and
        self.namespace).
        '''

        # Resolve the namespace, resulting in a dictionary containing only the
        # external variables that are needed by the code
        # Remove the variables in the specifiers dictionary first, as they are
        # not contained in the namespace
        identifiers = set(identifiers) - set(self.specifiers.keys())
        resolved_namespace = self.namespace.resolve_all(identifiers)

        return self.language.create_codeobj(name,
                                            code,
                                            resolved_namespace,
                                            self.specifiers,
                                            template,
                                            indices={'_neuron_idx':
                                                     Index('_neuron_idx',
                                                           iterate_all)})

    def create_state_updater(self):
        '''Create the `StateUpdater`.'''
        identifiers = self.equations.identifiers

        # For stochastic equations, we also need access to randn and _randn
        if not self.equations.stochastic_type is None:
            identifiers |= set(['randn', '_randn'])

        resolved_namespace = self.namespace.resolve_all(identifiers -
                                                        set(self.specifiers.keys()))

        self.abstract_code = self.method(self.equations,
                                         resolved_namespace,
                                         self.specifiers) 
        # The state updater might use exp
        identifiers |= set(['exp'])

        codeobj = self.create_codeobj("state updater",
                                      self.abstract_code,
                                      identifiers,
                                      self.language.template_state_update
                                      )

        state_updater = StateUpdater(self, codeobj,
                                     name=self.name + '_state_updater',
                                     when=(self.clock, 'groups'))
        return state_updater

    def runner(self, code, when=None, name=None):
        '''
        Returns a `CodeRunner` that runs abstract code in the groups namespace
        
        Parameters
        ----------
        code : str
            The abstract code to run.
        when : `Scheduler`, optional
            When to run, by default in the 'start' slot with the same clock as
            the group.
        name : str, optional
            A unique name, by default the name of the group appended with
            'runner_0', 'runner_1', etc.
        '''
        if when is None:  # TODO: make this better with default values
            when = Scheduler(clock=self.clock)
        else:
            raise NotImplementedError
        if name is None:
            if not hasattr(self, 'num_runners'):
                self.num_runners = 0
            name = self.name + '_runner_' + str(self.num_runners)
            self.num_runners += 1

        identifiers = get_identifiers(code)

        codeobj = self.create_codeobj("runner",
                                      code,
                                      identifiers,
                                      self.language.template_state_update,
                                      )
        runner = CodeRunner(codeobj, name=name, when=when)
        return runner

    def create_thresholder(self, threshold):
        '''Create the `Thresholder`'''
        if threshold is None:
            return None

        abstract_code = '_cond = ' + threshold
        identifiers = get_identifiers(threshold)

        codeobj = self.create_codeobj("thresholder",
                                      abstract_code,
                                      identifiers,
                                      self.language.template_threshold
                                      )
        thresholder = Thresholder(self, codeobj,
                                  name=self.name + '_thresholder',
                                  when=(self.clock, 'thresholds'))
        return thresholder

    def create_resetter(self, reset):
        '''Create the `Resetter`.'''
        if reset is None:
            return None

        abstract_code = reset
        identifiers = get_identifiers(reset)

        codeobj = self.create_codeobj("resetter",
                                      abstract_code,
                                      identifiers,
                                      self.language.template_reset,
                                      iterate_all=False
                                      )
        resetter = Resetter(self, codeobj,
                            name=self.name + '_resetter',
                            when=(self.clock, 'resets'))
        return resetter

    def create_specifiers(self):
        '''
        Create the specifiers dictionary for this `NeuronGroup`, containing
        entries for the equation variables and some standard entries.
        '''
        # Standard specifiers always present
        s = {'_num_neurons': ReadOnlyValue('_num_neurons', Unit(1), np.int, self.N),
             '_spikes' : AttributeValue('_spikes', Unit(1), np.int, self, 'spikes'),
             't': AttributeValue('t',  second, np.float64, self.clock, 't_'),
             'dt': AttributeValue('dt', second, np.float64, self.clock, 'dt_', constant=True)}

        # First add all the differential equations and parameters, because they
        # may be referred to by static equations
        for eq in self.equations.itervalues():
            if eq.eq_type in (DIFFERENTIAL_EQUATION, PARAMETER):
                array = self.arrays[eq.varname]
                constant = ('constant' in eq.flags)
                s.update({eq.varname: ArrayVariable(eq.varname,
                                                    eq.unit,
                                                    array.dtype,
                                                    array,
                                                    '_neuron_idx',
                                                    constant)})
        
        # Now go through the static equations in the correct order, i.e. each
        # static equation only depends on state variables (differential
        # equations or parameters) or *previous* static equations 
        for eq in self.equations.ordered:        
            if eq.eq_type == STATIC_EQUATION:
                # all external identifiers
                identifiers = eq.identifiers - self.equations.names - set(s.keys())
                resolved_namespace = self.namespace.resolve_all(identifiers)
                
                # because we are going through the equations that are already
                # ordered, subexpressions only need information about specifiers
                # that are already in the specifiers dictionary at this point.
                # To avoid circular references, we create a copy of the
                # specifiers dictionary that only contains weak references, so
                # it does not prevent them from being garbage collected
                specifiers_copy = {}
                for k, v in s.iteritems():
                    specifiers_copy[k] = weakref.proxy(v)
                s.update({eq.varname: Subexpression(eq.varname, eq.unit,
                                                    brian_prefs['core.default_scalar_dtype'],
                                                    str(eq.expr),
                                                    specifiers_copy,
                                                    resolved_namespace)})


        # Stochastic variables
        for xi in self.equations.stochastic_variables:
            s.update({xi: StochasticVariable(xi)})

        return s


if __name__ == '__main__':
    from pylab import *
    from brian2 import *
    from brian2.codegen.languages import *
    import time

    N = 10000
    tau = 10 * ms
    eqs = '''
    dV/dt = (2*volt-V)/tau_real : volt
    tau_real = 1 * tau : second # just to test that static equations work
    Vt : volt
    '''
    threshold = 'V>Vt'
    reset = 'V = 0*volt'    
    G = NeuronGroup(N, eqs,
                    threshold=threshold,
                    reset=reset,
                    language=CPPLanguage(),
                    method=euler,
                    # language=NumexprPythonLanguage(),
                    )
    print G.abstract_code
    G.refractory = 5 * ms
    Gmid = Subgroup(G, 40, 80)

    G.Vt = 1 * volt
    runner = G.runner('Vt = 1*volt-(t/second)*5*volt')

    G.V = linspace(0, 1, N) * volt

    statemon = StateMonitor(G, 'V', record=range(0, 10000, 2500))
    spikemon = SpikeMonitor(Gmid)

    start = time.time()
    run(1 * ms)
    print 'Initialise time:', time.time() - start
    start = time.time()
    run(99 * ms)
    print 'Runtime:', time.time() - start
    subplot(121)
    plot(statemon.t, statemon.V)
    subplot(122)
    plot(spikemon.t, spikemon.i, '.k')
    show()
