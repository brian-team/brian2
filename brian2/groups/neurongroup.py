'''
This model defines the `NeuronGroup`, the core of most simulations.
'''
import numpy as np
from numpy import array
import sympy

from brian2.equations.equations import (Equations, DIFFERENTIAL_EQUATION,
                                        STATIC_EQUATION, PARAMETER)
from brian2.equations.refractory import add_refractoriness
from brian2.equations.unitcheck import unit_from_expression
from brian2.stateupdaters.base import StateUpdateMethod
from brian2.codegen.languages import PythonLanguage
from brian2.memory import allocate_array
from brian2.core.preferences import brian_prefs
from brian2.core.base import BrianObject
from brian2.core.namespace import create_namespace
from brian2.core.specifiers import (ReadOnlyValue, AttributeValue, ArrayVariable,
                                    StochasticVariable, Subexpression)
from brian2.core.spikesource import SpikeSource
from brian2.core.scheduler import Scheduler
from brian2.utils.logger import get_logger
from brian2.units.allunits import second
from brian2.units.fundamentalunits import (Quantity, Unit, have_same_dimensions,
                                           DIMENSIONLESS)

from .group import Group, GroupCodeRunner

__all__ = ['NeuronGroup']

logger = get_logger(__name__)


class StateUpdater(GroupCodeRunner):
    '''
    The `GroupCodeRunner` that updates the state variables of a `NeuronGroup`
    at every timestep.
    '''
    def __init__(self, group, method):
        self.method_choice = method
        
        GroupCodeRunner.__init__(self, group,
                                       group.language.template_state_update,
                                       when=(group.clock, 'groups'),
                                       name=group.name+'_stateupdater*',
                                       check_units=False)        

        self.method = StateUpdateMethod.determine_stateupdater(self.group.equations,
                                                               self.group.namespace,
                                                               self.group.specifiers,
                                                               method)
    
    def update_abstract_code(self, additional_namespace):
        
        self.method = StateUpdateMethod.determine_stateupdater(self.group.equations,
                                                               self.group.namespace,
                                                               self.group.specifiers,
                                                               self.method_choice)

        # Update the not_refractory variable for the refractory period mechanism
        ref = self.group._refractory
        if ref is None:
            # No refractoriness
            self.abstract_code = ''
        elif isinstance(ref, Quantity):
            self.abstract_code = 'not_refractory = (t - lastspike) > %f\n' % ref
        else:
            namespace = dict(self.group.namespace)
            if additional_namespace is not None:
                namespace.update(additional_namespace[1])
            unit = unit_from_expression(ref, namespace, self.group.specifiers)
            if have_same_dimensions(unit, second):
                self.abstract_code = 'not_refractory = (t - lastspike) > %s\n' % ref
            elif have_same_dimensions(unit, Unit(1)):
                # boolean condition
                # we have to be a bit careful here, we can't just use the given
                # condition as it is, because we only want to *leave*
                # refractoriness, based on the condition
                self.abstract_code = 'not_refractory = bool(not_refractory) or not (%s)\n' % ref
            else:
                raise TypeError(('Refractory expression has to evaluate to a #'
                                 'timespan or a boolean value, expression'
                                 '"%s" has units %s instead') % (ref, unit))
        
        self.abstract_code += self.method(self.group.equations,
                                          self.group.namespace,
                                          self.group.specifiers)


class Thresholder(GroupCodeRunner):
    '''
    The `GroupCodeRunner` that applies the threshold condition to the state
    variables of a `NeuronGroup` at every timestep and sets its ``spikes``
    and ``refractory_until`` attributes.
    '''
    def __init__(self, group):
        GroupCodeRunner.__init__(self, group,
                                 group.language.template_threshold,
                                 when=(group.clock, 'thresholds'),
                                 name=group.name+'_thresholder*')
    
    def update_abstract_code(self, namespace):
        self.abstract_code = '_cond = ' + self.group.threshold
        
    def post_update(self, return_value):
        # Save the spikes in the NeuronGroup so others can use it
        self.group.spikes = return_value        


class Resetter(GroupCodeRunner):
    '''
    The `GroupCodeRunner` that applies the reset statement(s) to the state
    variables of neurons that have spiked in this timestep.
    '''
    def __init__(self, group):
        GroupCodeRunner.__init__(self, group,
                                 group.language.template_reset,
                                 when=(group.clock, 'resets'),
                                 name=group.name+'_resetter*',
                                 iterate_all=False)
    
    def update_abstract_code(self, namespace):
        self.abstract_code = self.group.reset


class NeuronGroup(BrianObject, Group, SpikeSource):
    '''
    A group of neurons.

    
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
    refractory : {str, `Quantity`}, optional
        Either the length of the refractory period (e.g. ``2*ms``), a string
        expression that evaluates to the length of the refractory period
        after each spike (e.g. ``'(1 + rand())*ms'``), or a string expression
        evaluating to a boolean value, given the condition under which the
        neuron stays refractory after a spike (e.g. ``'v > -20*mV'``)
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
    def __init__(self, N, equations, method=None,
                 threshold=None,
                 reset=None,
                 refractory=False,
                 namespace=None,
                 dtype=None, language=None,
                 clock=None, name='neurongroup*'):
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
            raise TypeError(('equations has to be a string or an Equations '
                             'object, is "%s" instead.') % type(equations))

        # Check flags
        equations.check_flags({DIFFERENTIAL_EQUATION: ('unless-refractory'),
                               PARAMETER: ('constant')})

        # add refractoriness
        equations = add_refractoriness(equations)
        self.equations = equations
        uses_refractoriness = len(equations) and any(['unless-refractory' in eq.flags
                                                      for eq in equations.itervalues()
                                                      if eq.type == DIFFERENTIAL_EQUATION])

        logger.debug("Creating NeuronGroup of size {self.N}, "
                     "equations {self.equations}.".format(self=self))

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

        #: The refractory condition or timespan
        self._refractory = refractory
        if uses_refractoriness and refractory is False:
            logger.warn('Model equations use the "unless-refractory" flag but '
                        'no refractory keyword was given.', 'no_refractory')

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

        # Set the refractoriness information
        self.lastspike = -np.inf*second
        self.not_refractory = True

    def __len__(self):
        '''
        Return number of neurons in the group.
        '''
        return self.N

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
            A unique name, if non is given the name of the group appended with
            'runner', 'runner_1', etc. will be used. If a name is given
            explicitly, it will be used as given (i.e. the group name will not
            be prepended automatically).
        '''
        if when is None:  # TODO: make this better with default values
            when = Scheduler(clock=self.clock)
        else:
            raise NotImplementedError

        if name is None:
            name = self.name + '_runner*'

        runner = GroupCodeRunner(self, self.language.template_state_update,
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
            if eq.type in (DIFFERENTIAL_EQUATION, PARAMETER):
                array = self.arrays[eq.varname]
                constant = ('constant' in eq.flags)
                s.update({eq.varname: ArrayVariable(eq.varname,
                                                    eq.unit,
                                                    array.dtype,
                                                    array,
                                                    '_neuron_idx',
                                                    constant)})
        
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
    
    def _repr_html_(self):
        text = [r'NeuronGroup "%s" with %d neurons.<br>' % (self.name, self.N)]
        text.append(r'<b>Model:</b><nr>')
        text.append(sympy.latex(self.equations))
        text.append(r'<b>Integration method:</b><br>')
        text.append(sympy.latex(self.state_updater.method)+'<br>')
        if self.threshold is not None:
            text.append(r'<b>Threshold condition:</b><br>')
            text.append('<code>%s</code><br>' % str(self.threshold))
            text.append('')
        if self.reset is not None:
            text.append(r'<b>Reset statement:</b><br>')            
            text.append(r'<code>%s</code><br>' % str(self.reset))
            text.append('')
                    
        return '\n'.join(text)
