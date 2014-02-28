'''
This model defines the `NeuronGroup`, the core of most simulations.
'''
import numpy as np
import sympy

from brian2.equations.equations import (Equations, DIFFERENTIAL_EQUATION,
                                        SUBEXPRESSION, PARAMETER)
from brian2.equations.refractory import add_refractoriness
from brian2.stateupdaters.base import StateUpdateMethod
from brian2.codegen.translation import analyse_identifiers
from brian2.codegen.codeobject import check_code_units
from brian2.core.variables import Variables
from brian2.core.spikesource import SpikeSource
from brian2.parsing.expressions import (parse_expression_unit,
                                        is_boolean_expression)
from brian2.utils.logger import get_logger
from brian2.utils.stringtools import get_identifiers
from brian2.units.allunits import second
from brian2.units.fundamentalunits import Quantity, Unit, have_same_dimensions


from .group import Group, CodeRunner, dtype_dictionary
from .subgroup import Subgroup


__all__ = ['NeuronGroup']

logger = get_logger(__name__)


class StateUpdater(CodeRunner):
    '''
    The `CodeRunner` that updates the state variables of a `NeuronGroup`
    at every timestep.
    '''
    def __init__(self, group, method):
        self.method_choice = method
        
        CodeRunner.__init__(self, group,
                            'stateupdate',
                            when=(group.clock, 'groups'),
                            name=group.name + '_stateupdater*',
                            check_units=False)

        # Don't do the check here for now since we don't have all the
        # information about functions yet
        # self.method = StateUpdateMethod.determine_stateupdater(self.group.equations,
        #                                                        self.group.variables,
        #                                                        method)

        # Generate the refractory code to catch errors in the refractoriness
        # formulation. However, do not fail on KeyErrors since the
        # refractoriness might refer to variables that don't exist yet
        try:
            self._get_refractory_code(run_namespace=None, level=1)
        except KeyError as ex:
            logger.debug('Namespace not complete (yet), ignoring: %s ' % str(ex),
                         'StateUpdater')

    def _get_refractory_code(self, run_namespace, level=0):
        ref = self.group._refractory
        if ref is False:
            # No refractoriness
            abstract_code = ''
        elif isinstance(ref, Quantity):
            abstract_code = 'not_refractory = (t - lastspike) > %f\n' % ref
        else:
            identifiers = get_identifiers(ref)
            variables = self.group.resolve_all(identifiers,
                                               run_namespace=run_namespace,
                                               level=level+1)
            unit = parse_expression_unit(str(ref), variables)
            if have_same_dimensions(unit, second):
                abstract_code = 'not_refractory = (t - lastspike) > %s\n' % ref
            elif have_same_dimensions(unit, Unit(1)):
                if not is_boolean_expression(str(ref), variables):
                    raise TypeError(('Refractory expression is dimensionless '
                                     'but not a boolean value. It needs to '
                                     'either evaluate to a timespan or to a '
                                     'boolean value.'))
                # boolean condition
                # we have to be a bit careful here, we can't just use the given
                # condition as it is, because we only want to *leave*
                # refractoriness, based on the condition
                abstract_code = 'not_refractory = not_refractory or not (%s)\n' % ref
            else:
                raise TypeError(('Refractory expression has to evaluate to a '
                                 'timespan or a boolean value, expression'
                                 '"%s" has units %s instead') % (ref, unit))
        return abstract_code

    def update_abstract_code(self, run_namespace=None, level=0):

        # Update the not_refractory variable for the refractory period mechanism
        self.abstract_code = self._get_refractory_code(run_namespace=run_namespace,
                                                       level=level+1)

        # Get the names used in the refractory code
        _, used_known, unknown = analyse_identifiers(self.abstract_code, self.group.variables,
                                                     recursive=True)

        # Get all names used in the equations (and always get "dt")
        names = self.group.equations.names
        external_names = self.group.equations.identifiers | set(['dt'])

        variables = self.group.resolve_all(used_known | unknown | names | external_names,
                                           run_namespace=run_namespace, level=level+1)

        # Since we did not necessarily no all the functions at creation time,
        # we might want to reconsider our numerical integration method
        self.method = StateUpdateMethod.determine_stateupdater(self.group.equations,
                                                               variables,
                                                               self.method_choice)
        self.abstract_code += self.method(self.group.equations, variables)


class Thresholder(CodeRunner):
    '''
    The `CodeRunner` that applies the threshold condition to the state
    variables of a `NeuronGroup` at every timestep and sets its ``spikes``
    and ``refractory_until`` attributes.
    '''
    def __init__(self, group):
        if group._refractory is False:
            template_kwds = {'_uses_refractory': False}
            needed_variables = []
        else:
            template_kwds = {'_uses_refractory': True}
            needed_variables=['t', 'not_refractory', 'lastspike']
        CodeRunner.__init__(self, group,
                            'threshold',
                            when=(group.clock, 'thresholds'),
                            name=group.name+'_thresholder*',
                            needed_variables=needed_variables,
                            template_kwds=template_kwds)

        # Check the abstract code for unit mismatches (only works if the
        # namespace is already complete)
        try:
            self.update_abstract_code(level=1)
            check_code_units(self.abstract_code, self.group)
        except KeyError:
            pass

    def update_abstract_code(self, run_namespace=None, level=0):
        code = self.group.threshold
        identifiers = get_identifiers(code)
        variables = self.group.resolve_all(identifiers,
                                           run_namespace=run_namespace,
                                           level=level+1)
        if not is_boolean_expression(self.group.threshold, variables):
            raise TypeError(('Threshold condition "%s" is not a boolean '
                             'expression') % self.group.threshold)
        if self.group._refractory is False:
            self.abstract_code = '_cond = %s' % self.group.threshold
        else:
            self.abstract_code = '_cond = (%s) and not_refractory' % self.group.threshold
        

class Resetter(CodeRunner):
    '''
    The `CodeRunner` that applies the reset statement(s) to the state
    variables of neurons that have spiked in this timestep.
    '''
    def __init__(self, group):
        CodeRunner.__init__(self, group,
                            'reset',
                            when=(group.clock, 'resets'),
                            name=group.name + '_resetter*',
                            override_conditional_write=['not_refractory'])

        # Check the abstract code for unit mismatches (only works if the
        # namespace is already complete)
        try:
            self.update_abstract_code(level=1)
            check_code_units(self.abstract_code, self.group)
        except KeyError:
            pass

    def update_abstract_code(self, run_namespace=None, level=0):
        self.abstract_code = self.group.reset


class NeuronGroup(Group, SpikeSource):
    '''
    A group of neurons.

    
    Parameters
    ----------
    N : int
        Number of neurons in the group.
    model : (str, `Equations`)
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
        The `numpy.dtype` that will be used to store the values, or a
        dictionary specifying the type for variable names. If a value is not
        provided for a variable (or no value is provided at all), the preference
        setting `core.default_scalar_dtype` is used.
    codeobj_class : class, optional
        The `CodeObject` class to run code with.
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
    def __init__(self, N, model, method=None,
                 threshold=None,
                 reset=None,
                 refractory=False,
                 namespace=None,
                 dtype=None,
                 clock=None, name='neurongroup*',
                 codeobj_class=None):
        Group.__init__(self, when=clock, name=name)

        self.codeobj_class = codeobj_class

        try:
            self._N = N = int(N)
        except ValueError:
            if isinstance(N, str):
                raise TypeError("First NeuronGroup argument should be size, not equations.")
            raise
        if N < 1:
            raise ValueError("NeuronGroup size should be at least 1, was " + str(N))

        self.start = 0
        self.stop = self._N

        ##### Prepare and validate equations
        if isinstance(model, basestring):
            model = Equations(model)
        if not isinstance(model, Equations):
            raise TypeError(('model has to be a string or an Equations '
                             'object, is "%s" instead.') % type(model))

        # Check flags
        model.check_flags({DIFFERENTIAL_EQUATION: ('unless refractory',),
                           PARAMETER: ('constant', 'scalar'),
                           SUBEXPRESSION: ('scalar',)})

        # add refractoriness
        if refractory is not False:
            model = add_refractoriness(model)
        self.equations = model
        uses_refractoriness = len(model) and any(['unless refractory' in eq.flags
                                                  for eq in model.itervalues()
                                                  if eq.type == DIFFERENTIAL_EQUATION])

        logger.debug("Creating NeuronGroup of size {self._N}, "
                     "equations {self.equations}.".format(self=self))

        if namespace is None:
            namespace = {}
        #: The group-specific namespace
        self.namespace = namespace

        # Setup variables
        self._create_variables(dtype)

        # All of the following will be created in before_run
        
        #: The threshold condition
        self.threshold = threshold
        
        #: The reset statement(s)
        self.reset = reset

        #: The refractory condition or timespan
        self._refractory = refractory
        if uses_refractoriness and refractory is False:
            logger.warn('Model equations use the "unless refractory" flag but '
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

        # We try to run a before_run already now. This might fail because of an
        # incomplete namespace but if the namespace is already complete we
        # can spot unit errors in the equation already here.
        try:
            self.before_run(None)
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

        if refractory is not False:
            # Set the refractoriness information
            self.variables['lastspike'].set_value(-np.inf*second)
            self.variables['not_refractory'].set_value(True)

        # Activate name attribute access
        self._enable_group_attributes()


    def __len__(self):
        '''
        Return number of neurons in the group.
        '''
        return self.N

    @property
    def spikes(self):
        '''
        The spikes returned by the most recent thresholding operation.
        '''
        # Note that we have to directly access the ArrayVariable object here
        # instead of using the Group mechanism by accessing self._spikespace
        # Using the latter would cut _spikespace to the length of the group
        spikespace = self.variables['_spikespace'].get_value()
        return spikespace[:spikespace[-1]]

    def __getitem__(self, item):
        if not isinstance(item, slice):
            raise TypeError('Subgroups can only be constructed using slicing syntax')
        start, stop, step = item.indices(self._N)
        if step != 1:
            raise IndexError('Subgroups have to be contiguous')
        if start >= stop:
            raise IndexError('Illegal start/end values for subgroup, %d>=%d' %
                             (start, stop))

        return Subgroup(self, start, stop)

    def _create_variables(self, dtype=None):
        '''
        Create the variables dictionary for this `NeuronGroup`, containing
        entries for the equation variables and some standard entries.
        '''
        self.variables = Variables(self)
        self.variables.add_clock_variables(self.clock)
        self.variables.add_constant('N', Unit(1), self._N)

        dtype = dtype_dictionary(dtype)

        # Standard variables always present
        self.variables.add_array('_spikespace', unit=Unit(1), size=self._N+1,
                                 dtype=np.int32, constant=False)
        # Add the special variable "i" which can be used to refer to the neuron index
        self.variables.add_arange('i', size=self._N, constant=True,
                                  read_only=True)

        for eq in self.equations.itervalues():
            if eq.type in (DIFFERENTIAL_EQUATION, PARAMETER):
                constant = 'constant' in eq.flags
                scalar = 'scalar' in eq.flags
                size = 1 if scalar else self._N
                index = '0' if scalar else None
                self.variables.add_array(eq.varname, size=size,
                                         unit=eq.unit, dtype=dtype[eq.varname],
                                         constant=constant, is_bool=eq.is_bool,
                                         scalar=scalar,
                                         index=index)
            elif eq.type == SUBEXPRESSION:
                self.variables.add_subexpression(eq.varname, unit=eq.unit,
                                                 expr=str(eq.expr),
                                                 is_bool=eq.is_bool,
                                                 scalar='scalar' in eq.flags)
            else:
                raise AssertionError('Unknown type of equation: ' + eq.eq_type)

        # Add the conditional-write attribute for variables with the
        # "unless refractory" flag
        for eq in self.equations.itervalues():
            if eq.type == DIFFERENTIAL_EQUATION and 'unless refractory' in eq.flags:
                not_refractory_var = self.variables['not_refractory']
                self.variables[eq.varname].set_conditional_write(not_refractory_var)

        # Stochastic variables
        for xi in self.equations.stochastic_variables:
            self.variables.add_auxiliary_variable(xi, unit=second**-0.5)

        # Check scalar subexpressions
        for eq in self.equations.itervalues():
            if eq.type == SUBEXPRESSION and 'scalar' in eq.flags:
                var = self.variables[eq.varname]
                for identifier in var.identifiers:
                    if identifier in self.variables:
                        if not self.variables[identifier].scalar:
                            raise SyntaxError(('Scalar subexpression %s refers '
                                               'to non-scalar variable %s.')
                                              % (eq.varname, identifier))


    def before_run(self, run_namespace=None, level=0):
        # Check units
        self.equations.check_units(self, run_namespace=run_namespace,
                                   level=level+1)
    
    def _repr_html_(self):
        text = [r'NeuronGroup "%s" with %d neurons.<br>' % (self.name, self._N)]
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
