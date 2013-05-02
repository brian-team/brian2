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
from brian2.core.namespace import create_namespace
from brian2.core.specifiers import (ReadOnlyValue, AttributeValue, ArrayVariable,
                                    StochasticVariable, Subexpression, Index)
from brian2.core.spikesource import SpikeSource
from brian2.core.scheduler import Scheduler
from brian2.utils.logger import get_logger
from brian2.groups.group import Group
from brian2.units.allunits import second
from brian2.units.fundamentalunits import Unit, Quantity
from brian2.stateupdaters.exact import linear
from brian2.codegen.translation import analyse_identifiers
from brian2.equations.unitcheck import check_units_statements

__all__ = ['NeuronGroup']

logger = get_logger(__name__)

def _create_codeobj(group, name, code, additional_namespace=None,
                    template=None, iterate_all=True, check_units=True):
    ''' A little helper function to reduce the amount of repetition when
    calling the language's _create_codeobj (always pass self.specifiers and
    self.namespace + additional namespace).
    '''

    if check_units:
        # Resolve the namespace, resulting in a dictionary containing only the
        # external variables that are needed by the code -- kepp the units for
        # the unit checks 
        _, _, unknown = analyse_identifiers(code, group.specifiers.keys())
        resolved_namespace = group.namespace.resolve_all(unknown,
                                                         additional_namespace,
                                                         strip_units=False)
    
        check_units_statements(code, resolved_namespace, group.specifiers)

    # Get the namespace without units
    _, _, unknown = analyse_identifiers(code, group.specifiers.keys())
    resolved_namespace = group.namespace.resolve_all(unknown,
                                                     additional_namespace)
    return group.language.create_codeobj(name,
                                         code,
                                         resolved_namespace,
                                         group.specifiers,
                                         template,
                                         indices={'_neuron_idx':
                                                  Index('_neuron_idx',
                                                        iterate_all)})


class NeuronGroupCodeRunner(BrianObject):
    '''
    A `CodeRunner` for use in `NeuronGroup`. Keeps a reference to the
    `NeuronGroup`.
    '''
    def __init__(self, group, template, code=None, iterate_all=True,
                 when=None, name=None, check_units=True):
        BrianObject.__init__(self, when=when, name=name)
        self.group = weakref.proxy(group)
        self.template = template
        self.abstract_code = code
        self.iterate_all = iterate_all
        self.check_units = check_units
        # Try to generate the abstract code and the codeobject without any
        # additional namespace. This might work in situations where the
        # namespace is completely defined in the NeuronGroup. In this case,
        # we might spot parsing or unit errors already now and don't have to
        # wait until the run call. We want to ignore KeyErrors, though, because
        # they possibly result from an incomplete namespace, which is still ok
        # at this time.
        try:
            self.pre_run(None)
        except KeyError:
            pass 
    
    def update_abstract_code(self):
        pass
    
    def pre_run(self, namespace):
        self.update_abstract_code()
        self.codeobj = _create_codeobj(self.group, self.name,
                                       self.abstract_code,
                                       additional_namespace=namespace,
                                       template=self.template,
                                       iterate_all=self.iterate_all,
                                       check_units=self.check_units
                                       )
    
    def pre_update(self):
        pass
    
    def update(self, **kwds):
        self.pre_update()
        return_value = self.codeobj(**kwds)
        self.post_update(return_value)    

    def post_update(self, return_value):
        pass


class StateUpdater(NeuronGroupCodeRunner):
    def __init__(self, group, method):
        self.method_choice = method
        
        NeuronGroupCodeRunner.__init__(self, group,
                                       group.language.template_state_update,
                                       when=(group.clock, 'groups'),
                                       name=group.name + '_stateupdater',
                                       check_units=False)        

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
    
    def pre_update(self):
        self.group.is_active_[:] = self.group.clock.t_ >= self.group.refractory_until_


class Thresholder(NeuronGroupCodeRunner):
    def __init__(self, group):
        NeuronGroupCodeRunner.__init__(self, group,
                                       group.language.template_threshold,
                                       when=(group.clock, 'thresholds'),
                                       name=group.name + '_thresholder')
    
    def update_abstract_code(self):
        self.abstract_code = '_cond = ' + self.group.threshold
        
    def post_update(self, return_value):
        spikes = return_value
        # Save the spikes in the NeuronGroup so others can use it
        self.group.spikes = spikes
        self.group.refractory_until_[spikes] = self.group.clock.t_ + self.group.refractory_[spikes]


class Resetter(NeuronGroupCodeRunner):
    def __init__(self, group):
        NeuronGroupCodeRunner.__init__(self, group,
                                       group.language.template_reset,
                                       when=(group.clock, 'resets'),
                                       name=group.name + '_resetter',
                                       iterate_all=False)
    
    def update_abstract_code(self):
        self.abstract_code = self.group.reset


class NeuronGroup(BrianObject, Group, SpikeSource):
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
        self.arrays = self._allocate_memory(dtype=dtype)

        #: The array of spikes from the most recent threshold operation
        self.spikes = array([], dtype=int)

        # Setup the namespace
        self.namespace = create_namespace(self.N, namespace)

        # Setup specifiers
        self.specifiers = self._create_specifiers()

        # Code generation (TODO: this should be refactored and modularised)
        # Temporary, set default language to Python
        if language is None:
            language = PythonLanguage()
        self.language = language

        # All of the following will be created in pre_run
        
        #: The threshold condition
        self.threshold = threshold
        
        #: The reset statement(s)
        self.reset = reset
        
        #: The state update method selected by the user
        self.method_choice = method
        
        #: Performs thresholding step, sets the value of `spikes`
        self.thresholder = None
        if self.threshold is not None:
            self.thresholder = Thresholder(self)
            
        #: Resets neurons which have spiked (`spikes`)
        self.resetter = None
        if self.reset is not None:
            self.resetter = Resetter(self)

        # We try to run a pre_run already now. This might fail because of an
        # incomplete namespace but if the namespace is already complete we
        # can spot unit or syntax errors already here, at creation time.
        try:
            self.pre_run(None)
        except KeyError:
            pass

        #: Performs numerical integration step
        self.state_updater = StateUpdater(self, method)

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


    def _allocate_memory(self, dtype=None):
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

        runner = NeuronGroupCodeRunner(self,
                                       self.language.template_state_update,
                                       code=code, name=name, when=when)
        return runner

    def _create_specifiers(self):
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
            elif eq.eq_type == STATIC_EQUATION:
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

    def pre_run(self, namespace):

        # Update the namespace information in the specifiers in case the
        # namespace was not specified explicitly defined at creation time
        # Note that values in the explicit namespace might still change
        # between runs, but the Subexpression stores a reference to 
        # self.namespace so these changes are taken into account automatically
        if not self.namespace.is_explicit:
            for spec in self.specifiers.itervalues():
                if isinstance(spec, Subexpression):
                    spec.additional_namespace = namespace

        # Check units
        self.equations.check_units(self.namespace, self.specifiers,
                                   namespace)


if __name__ == '__main__':
    from pylab import *
    from brian2 import *
    from brian2.codegen.languages import *
    import time
    
    #BrianLogger.log_level_debug()

    N = 10000
    tau = 10*ms
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
                    language=PythonLanguage(),
                    method=linear,
                    # language=NumexprPythonLanguage(),
                    )
    G.refractory = 5 * ms
    Gmid = Subgroup(G, 40, 80)

    G.Vt = 1 * volt
    runner = G.runner('Vt = 1*volt-(t/second)*5*volt')

    G.V = linspace(0, 1, N) * volt
    statemon = StateMonitor(G, 'V', record=range(0, 10000, 2500))
    spikemon = SpikeMonitor(Gmid)
    start = time.time()
    run(50 * ms)
    
    print 'Initialise time:', time.time() - start
    start = time.time()
    tau = 20*ms # Change the value, should be taken into account    
    run(50 * ms)
    print 'Runtime:', time.time() - start
    subplot(121)
    plot(statemon.t, statemon.V)
    subplot(122)
    plot(spikemon.t, spikemon.i, '.k')
    show()
