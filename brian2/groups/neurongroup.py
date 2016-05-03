'''
This model defines the `NeuronGroup`, the core of most simulations.
'''
import string

import numpy as np
import sympy
from pyparsing import Word

from brian2.equations.equations import (Equations, DIFFERENTIAL_EQUATION,
                                        SUBEXPRESSION, PARAMETER)
from brian2.equations.refractory import add_refractoriness
from brian2.stateupdaters.base import StateUpdateMethod
from brian2.codegen.translation import analyse_identifiers
from brian2.codegen.codeobject import check_code_units
from brian2.core.variables import (Variables, LinkedVariable,
                                   DynamicArrayVariable, Subexpression)
from brian2.core.spikesource import SpikeSource
from brian2.parsing.expressions import (parse_expression_unit,
                                        is_boolean_expression)
from brian2.utils.logger import get_logger
from brian2.utils.stringtools import get_identifiers
from brian2.units.allunits import second
from brian2.units.fundamentalunits import (Quantity, Unit,
                                           have_same_dimensions,
                                           DimensionMismatchError,
                                           fail_for_dimension_mismatch)


from .group import Group, CodeRunner, get_dtype
from .subgroup import Subgroup


__all__ = ['NeuronGroup']

logger = get_logger(__name__)


IDENTIFIER = Word(string.ascii_letters + '_',
                  string.ascii_letters + string.digits + '_').setResultsName('identifier')

def _valid_event_name(event_name):
    '''
    Helper function to check whether a name is a valid name for an event.

    Parameters
    ----------
    event_name : str
        The name to check

    Returns
    -------
    is_valid : bool
        Whether the given name is valid
    '''
    parse_result = list(IDENTIFIER.scanString(event_name))

    # parse_result[0][0][0] refers to the matched string -- this should be the
    # full identifier, if not it is an illegal identifier like "3foo" which only
    # matched on "foo"
    return len(parse_result) == 1 and parse_result[0][0][0] == event_name

def _guess_membrane_potential(equations):
    '''
    Little helper function to guess which variable represents the membrane
    potential. This follows the same logic as in Brian1 but is only used to
    give a suggestion in the error message when a Brian1-style syntax is used
    for threshold or reset.
    '''
    if len(equations) == 1:
        return equations.keys()[0]
    for name, eq in equations.iteritems():
        if name in ['V', 'v', 'Vm', 'vm']:
            return name

    # nothing found
    return None


class StateUpdater(CodeRunner):
    '''
    The `CodeRunner` that updates the state variables of a `NeuronGroup`
    at every timestep.
    '''
    def __init__(self, group, method):
        self.method_choice = method
        CodeRunner.__init__(self, group,
                            'stateupdate',
                            code='',  # will be set in update_abstract_code
                            clock=group.clock,
                            when='groups',
                            order=group.order,
                            name=group.name + '_stateupdater*',
                            check_units=False,
                            generate_empty_code=False)

    def _get_refractory_code(self, run_namespace):
        ref = self.group._refractory
        if ref is False:
            # No refractoriness
            abstract_code = ''
        elif isinstance(ref, Quantity):
            fail_for_dimension_mismatch(ref, second, ('Refractory period has to '
                                                      'be specified in units '
                                                      'of seconds but got '
                                                      '{value}'),
                                        value=ref)

            abstract_code = 'not_refractory = (t - lastspike) > %f\n' % ref
        else:
            identifiers = get_identifiers(ref)
            variables = self.group.resolve_all(identifiers,
                                               identifiers,
                                               run_namespace=run_namespace)
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

    def update_abstract_code(self, run_namespace):

        # Update the not_refractory variable for the refractory period mechanism
        self.abstract_code = self._get_refractory_code(run_namespace=run_namespace)

        # Get the names used in the refractory code
        _, used_known, unknown = analyse_identifiers(self.abstract_code, self.group.variables,
                                                     recursive=True)

        # Get all names used in the equations (and always get "dt")
        names = self.group.equations.names
        external_names = self.group.equations.identifiers | {'dt'}

        variables = self.group.resolve_all(used_known | unknown | names | external_names,
                                           # we don't need to raise any warnings
                                           # for the user here, warnings will
                                           # be raised in create_runner_codeobj
                                           set(),
                                           run_namespace=run_namespace)
        if len(self.group.equations.diff_eq_names) > 0:
            self.abstract_code += StateUpdateMethod.apply_stateupdater(self.group.equations,
                                                                       variables,
                                                                       self.method_choice,
                                                                       group_name=self.group.name)
        user_code = '\n'.join(['{var} = {expr}'.format(var=var, expr=expr)
                               for var, expr in
                               self.group.equations.get_substituted_expressions(variables)])
        self.user_code = user_code


class Thresholder(CodeRunner):
    '''
    The `CodeRunner` that applies the threshold condition to the state
    variables of a `NeuronGroup` at every timestep and sets its ``spikes``
    and ``refractory_until`` attributes.
    '''
    def __init__(self, group, when='thresholds', event='spike'):
        self.event = event
        if group._refractory is False or event != 'spike':
            template_kwds = {'_uses_refractory': False}
            needed_variables = []
        else:
            template_kwds = {'_uses_refractory': True}
            needed_variables=['t', 'not_refractory', 'lastspike']
        # Since this now works for general events not only spikes, we have to
        # pass the information about which variable to use to the template,
        # it can not longer simply refer to "_spikespace"
        eventspace_name = '_{}space'.format(event)
        template_kwds['eventspace_variable'] = group.variables[eventspace_name]
        needed_variables.append(eventspace_name)
        self.variables = Variables(self)
        self.variables.add_auxiliary_variable('_cond', unit=Unit(1),
                                              dtype=np.bool)
        CodeRunner.__init__(self, group,
                            'threshold',
                            code='',  # will be set in update_abstract_code
                            clock=group.clock,
                            when=when,
                            order=group.order,
                            name=group.name+'_thresholder*',
                            needed_variables=needed_variables,
                            template_kwds=template_kwds)

    def update_abstract_code(self, run_namespace):
        code = self.group.events[self.event]
        # Raise a useful error message when the user used a Brian1 syntax
        if not isinstance(code, basestring):
            if isinstance(code, Quantity):
                t = 'a quantity'
            else:
                t = '%s' % type(code)
            error_msg = 'Threshold condition has to be a string, not %s.' % t
            if self.event == 'spike':
                try:
                    vm_var = _guess_membrane_potential(self.group.equations)
                except AttributeError:  # not a group with equations...
                    vm_var = None
                if vm_var is not None:
                    error_msg += " Probably you intended to use '%s > ...'?" % vm_var
            raise TypeError(error_msg)

        self.user_code = '_cond = ' + code

        identifiers = get_identifiers(code)
        variables = self.group.resolve_all(identifiers,
                                           identifiers,
                                           run_namespace=run_namespace)
        if not is_boolean_expression(code, variables):
            raise TypeError(('Threshold condition "%s" is not a boolean '
                             'expression') % code)
        if self.group._refractory is False or self.event != 'spike':
            self.abstract_code = '_cond = %s' % code
        else:
            self.abstract_code = '_cond = (%s) and not_refractory' % code


class Resetter(CodeRunner):
    '''
    The `CodeRunner` that applies the reset statement(s) to the state
    variables of neurons that have spiked in this timestep.
    '''
    def __init__(self, group, when='resets', order=None, event='spike'):
        self.event = event
        # Since this now works for general events not only spikes, we have to
        # pass the information about which variable to use to the template,
        # it can not longer simply refer to "_spikespace"
        eventspace_name = '_{}space'.format(event)
        template_kwds = {'eventspace_variable': group.variables[eventspace_name]}
        needed_variables= [eventspace_name]
        order = order if order is not None else group.order
        CodeRunner.__init__(self, group,
                            'reset',
                            code='',  # will be set in update_abstract_code
                            clock=group.clock,
                            when=when,
                            order=order,
                            name=group.name + '_resetter*',
                            override_conditional_write=['not_refractory'],
                            needed_variables=needed_variables,
                            template_kwds=template_kwds)

    def update_abstract_code(self, run_namespace):
        code = self.group.event_codes[self.event]
        # Raise a useful error message when the user used a Brian1 syntax
        if not isinstance(code, basestring):
            if isinstance(code, Quantity):
                t = 'a quantity'
            else:
                t = '%s' % type(code)
            error_msg = 'Reset statement has to be a string, not %s.' % t
            if self.event == 'spike':
                vm_var = _guess_membrane_potential(self.group.equations)
                if vm_var is not None:
                    error_msg += " Probably you intended to use '%s = ...'?" % vm_var
            raise TypeError(error_msg)

        self.abstract_code = code


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
    events : dict, optional
        User-defined events in addition to the "spike" event defined by the
        ``threshold``. Has to be a mapping of strings (the event name) to
         strings (the condition) that will be checked.
    namespace: dict, optional
        A dictionary mapping variable/function names to the respective objects.
        If no `namespace` is given, the "implicit" namespace, consisting of
        the local and global namespace surrounding the creation of the class,
        is used.
    dtype : (`dtype`, `dict`), optional
        The `numpy.dtype` that will be used to store the values, or a
        dictionary specifying the type for variable names. If a value is not
        provided for a variable (or no value is provided at all), the preference
        setting `core.default_float_dtype` is used.
    codeobj_class : class, optional
        The `CodeObject` class to run code with.
    dt : `Quantity`, optional
        The time step to be used for the simulation. Cannot be combined with
        the `clock` argument.
    clock : `Clock`, optional
        The update clock to be used. If neither a clock, nor the `dt` argument
        is specified, the `defaultclock` will be used.
    order : int, optional
        The priority of of this group for operations occurring at the same time
        step and in the same scheduling slot. Defaults to 0.
    name : str, optional
        A unique name for the group, otherwise use ``neurongroup_0``, etc.
        
    Notes
    -----
    `NeuronGroup` contains a `StateUpdater`, `Thresholder` and `Resetter`, and
    these are run at the 'groups', 'thresholds' and 'resets' slots (i.e. the
    values of their `when` attribute take these values). The `order`
    attribute will be passed down to the contained objects but can be set
    individually by setting the `order` attribute of the `state_updater`,
    `thresholder` and `resetter` attributes, respectively.
    '''
    add_to_magic_network = True

    def __init__(self, N, model,
                 method=('linear', 'euler', 'heun'),
                 threshold=None,
                 reset=None,
                 refractory=False,
                 events=None,
                 namespace=None,
                 dtype=None,
                 dt=None,
                 clock=None,
                 order=0,
                 name='neurongroup*',
                 codeobj_class=None):
        Group.__init__(self, dt=dt, clock=clock, when='start', order=order,
                       name=name)

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
                           PARAMETER: ('constant', 'shared', 'linked'),
                           SUBEXPRESSION: ('shared',)})

        # add refractoriness
        if refractory is not False:
            model = add_refractoriness(model)
        self.equations = model
        uses_refractoriness = len(model) and any(['unless refractory' in eq.flags
                                                  for eq in model.itervalues()
                                                  if eq.type == DIFFERENTIAL_EQUATION])
        self._linked_variables = set()
        logger.diagnostic("Creating NeuronGroup of size {self._N}, "
                          "equations {self.equations}.".format(self=self))

        if namespace is None:
            namespace = {}
        #: The group-specific namespace
        self.namespace = namespace

        # All of the following will be created in before_run

        #: The refractory condition or timespan
        self._refractory = refractory
        if uses_refractoriness and refractory is False:
            logger.warn('Model equations use the "unless refractory" flag but '
                        'no refractory keyword was given.', 'no_refractory')

        #: The state update method selected by the user
        self.method_choice = method

        if events is None:
            events = {}

        if threshold is not None:
            if 'spike' in events:
                raise ValueError(("The NeuronGroup defines both a threshold "
                                  "and a 'spike' event"))
            events['spike'] = threshold

        # Setup variables
        # Since we have to create _spikespace and possibly other "eventspace"
        # variables, we pass the supported events
        self._create_variables(dtype, events=events.keys())

        #: Events supported by this group
        self.events = events

        #: Code that is triggered on events (e.g. reset)
        self.event_codes = {}

        #: Checks the spike threshold (or abitrary user-defined events)
        self.thresholder = {}

        #: Reset neurons which have spiked (or perform arbitrary actions for
        #: user-defined events)
        self.resetter = {}

        for event_name in events.iterkeys():
            if not isinstance(event_name, basestring):
                raise TypeError(('Keys in the "events" dictionary have to be '
                                 'strings, not type %s.') % type(event_name))
            if not _valid_event_name(event_name):
                raise TypeError(("The name '%s' cannot be used as an event "
                                 "name.") % event_name)
            # By default, user-defined events are checked after the threshold
            when = 'thresholds' if event_name == 'spike' else 'after_thresholds'
            # creating a Thresholder will take care of checking the validity
            # of the condition
            thresholder = Thresholder(self, event=event_name, when=when)
            self.thresholder[event_name] = thresholder
            self.contained_objects.append(thresholder)

        if reset is not None:
            self.run_on_event('spike', reset, when='resets')

        #: Performs numerical integration step
        self.state_updater = StateUpdater(self, method)

        # Creation of contained_objects that do the work
        self.contained_objects.append(self.state_updater)

        if refractory is not False:
            # Set the refractoriness information
            self.variables['lastspike'].set_value(-np.inf*second)
            self.variables['not_refractory'].set_value(True)

        # Activate name attribute access
        self._enable_group_attributes()

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

    def state(self, name, use_units=True, level=0):
        try:
            return Group.state(self, name, use_units=use_units, level=level+1)
        except KeyError as ex:
            if name in self._linked_variables:
                raise TypeError(('Link target for variable %s has not been '
                                 'set.') % name)
            else:
                raise ex

    def run_on_event(self, event, code, when='after_resets', order=None):
        '''
        Run code triggered by a custom-defined event (see `NeuronGroup`
        documentation for the specification of events).The created `Resetter`
        object will be automatically added to the group, it therefore does not
        need to be added to the network manually. However, a reference to the
        object will be returned, which can be used to later remove it from the
        group or to set it to inactive.

        Parameters
        ----------
        event : str
            The name of the event that should trigger the code
        code : str
            The code that should be executed
        when : str, optional
            The scheduling slot that should be used to execute the code.
            Defaults to `'after_resets'`.
        order : int, optional
            The order for operations in the same scheduling slot. Defaults to
            the order of the `NeuronGroup`.

        Returns
        -------
        obj : `Resetter`
            A reference to the object that will be run.
        '''
        if event not in self.events:
            error_message = "Unknown event '%s'." % event
            if event == 'spike':
                error_message += ' Did you forget to define a threshold?'
            raise ValueError(error_message)
        if event in self.resetter:
            raise ValueError(("Cannot add code for event '%s', code for this "
                              "event has already been added.") % event)
        self.event_codes[event] = code
        resetter = Resetter(self, when=when, order=order, event=event)
        self.resetter[event] = resetter
        self.contained_objects.append(resetter)

        return resetter

    def set_event_schedule(self, event, when='after_thresholds', order=None):
        '''
        Change the scheduling slot for checking the condition of an event.

        Parameters
        ----------
        event : str
            The name of the event for which the scheduling should be changed
        when : str, optional
            The scheduling slot that should be used to check the condition.
            Defaults to `'after_thresholds'`.
        order : int, optional
            The order for operations in the same scheduling slot. Defaults to
            the order of the `NeuronGroup`.
        '''
        if event not in self.thresholder:
            raise ValueError("Unknown event '%s'." % event)
        order = order if order is not None else self.order
        self.thresholder[event].when = when
        self.thresholder[event].order = order

    def __setattr__(self, key, value):
        # attribute access is switched off until this attribute is created by
        # _enable_group_attributes
        if not hasattr(self, '_group_attribute_access_active') or key in self.__dict__:
            object.__setattr__(self, key, value)
        elif key in self._linked_variables:
            if not isinstance(value, LinkedVariable):
                raise ValueError(('Cannot set a linked variable directly, link '
                                  'it to another variable using "linked_var".'))
            linked_var = value.variable
            
            if isinstance(linked_var, DynamicArrayVariable):
                raise NotImplementedError(('Linking to variable %s is not '
                                           'supported, can only link to '
                                           'state variables of fixed '
                                           'size.') % linked_var.name)

            eq = self.equations[key]
            if eq.unit != linked_var.unit:
                raise DimensionMismatchError(('Unit of variable %s does not '
                                              'match its link target %s') % (key,
                                                                             linked_var.name))

            if not isinstance(linked_var, Subexpression):
                var_length = len(linked_var)
            else:
                var_length = len(linked_var.owner)

            if value.index is not None:
                try:
                    index_array = np.asarray(value.index)
                    if not np.issubsctype(index_array.dtype, np.int):
                        raise TypeError()
                except TypeError:
                    raise TypeError(('The index for a linked variable has '
                                     'to be an integer array'))
                size = len(index_array)
                source_index = value.group.variables.indices[value.name]
                if source_index not in ('_idx', '0'):
                    # we are indexing into an already indexed variable,
                    # calculate the indexing into the target variable
                    index_array = value.group.variables[source_index].get_value()[index_array]

                if not index_array.ndim == 1 or size != len(self):
                    raise TypeError(('Index array for linked variable %s '
                                     'has to be a one-dimensional array of '
                                     'length %d, but has shape '
                                     '%s') % (key,
                                              len(self),
                                              str(index_array.shape)))
                if min(index_array) < 0 or max(index_array) >= var_length:
                    raise ValueError('Index array for linked variable %s '
                                     'contains values outside of the valid '
                                     'range [0, %d[' % (key,
                                                        var_length))
                self.variables.add_array('_%s_indices' % key, unit=Unit(1),
                                         size=size, dtype=index_array.dtype,
                                         constant=True, read_only=True,
                                         values=index_array)
                index = '_%s_indices' % key
            else:
                if linked_var.scalar or (var_length == 1 and self._N != 1):
                    index = '0'
                else:
                    index = value.group.variables.indices[value.name]
                    if index == '_idx':
                        target_length = var_length
                    else:
                        target_length = len(value.group.variables[index])
                        # we need a name for the index that does not clash with
                        # other names and a reference to the index
                        new_index = '_' + value.name + '_index_' + index
                        self.variables.add_reference(new_index,
                                                     value.group,
                                                     index)
                        index = new_index

                    if len(self) != target_length:
                        raise ValueError(('Cannot link variable %s to %s, the size of '
                                          'the target group does not match '
                                          '(%d != %d). You can provide an indexing '
                                          'scheme with the "index" keyword to link '
                                          'groups with different sizes') % (key,
                                                           linked_var.name,
                                                           len(self),
                                                           target_length))

            self.variables.add_reference(key,
                                         value.group,
                                         value.name,
                                         index=index)
            log_msg = ('Setting {target}.{targetvar} as a link to '
                       '{source}.{sourcevar}').format(target=self.name,
                                                      targetvar=key,
                                                      source=value.variable.owner.name,
                                                      sourcevar=value.variable.name)
            if index is not None:
                log_msg += '(using "{index}" as index variable)'.format(index=index)
            logger.diagnostic(log_msg)
        else:
            if isinstance(value, LinkedVariable):
                raise TypeError(('Cannot link variable %s, it has to be marked '
                                 'as a linked variable with "(linked)" in the '
                                 'model equations.') % key)
            else:
                Group.__setattr__(self, key, value, level=1)

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

    def _create_variables(self, user_dtype, events):
        '''
        Create the variables dictionary for this `NeuronGroup`, containing
        entries for the equation variables and some standard entries.
        '''
        self.variables = Variables(self)
        self.variables.add_constant('N', Unit(1), self._N)

        # Standard variables always present
        for event in events:
            self.variables.add_array('_{}space'.format(event), unit=Unit(1),
                                     size=self._N+1, dtype=np.int32,
                                     constant=False)
        # Add the special variable "i" which can be used to refer to the neuron index
        self.variables.add_arange('i', size=self._N, constant=True,
                                  read_only=True)
        # Add the clock variables
        self.variables.create_clock_variables(self._clock)

        for eq in self.equations.itervalues():
            dtype = get_dtype(eq, user_dtype)

            if eq.type in (DIFFERENTIAL_EQUATION, PARAMETER):
                if 'linked' in eq.flags:
                    # 'linked' cannot be combined with other flags
                    if not len(eq.flags) == 1:
                        raise SyntaxError(('The "linked" flag cannot be '
                                           'combined with other flags'))
                    self._linked_variables.add(eq.varname)
                else:
                    constant = 'constant' in eq.flags
                    shared = 'shared' in eq.flags
                    size = 1 if shared else self._N
                    self.variables.add_array(eq.varname, size=size,
                                             unit=eq.unit, dtype=dtype,
                                             constant=constant,
                                             scalar=shared)
            elif eq.type == SUBEXPRESSION:
                self.variables.add_subexpression(eq.varname, unit=eq.unit,
                                                 expr=str(eq.expr),
                                                 dtype=dtype,
                                                 scalar='shared' in eq.flags)
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
            if eq.type == SUBEXPRESSION and 'shared' in eq.flags:
                var = self.variables[eq.varname]
                for identifier in var.identifiers:
                    if identifier in self.variables:
                        if not self.variables[identifier].scalar:
                            raise SyntaxError(('Shared subexpression %s refers '
                                               'to non-shared variable %s.')
                                              % (eq.varname, identifier))

    def before_run(self, run_namespace=None):
        # Check units
        self.equations.check_units(self, run_namespace=run_namespace)

    def _repr_html_(self):
        text = [r'NeuronGroup "%s" with %d neurons.<br>' % (self.name, self._N)]
        text.append(r'<b>Model:</b><nr>')
        text.append(sympy.latex(self.equations))

        def add_event_to_text(event):
            if event=='spike':
                event_header = 'Spiking behaviour'
                event_condition = 'Threshold condition'
                event_code = 'Reset statement(s)'
            else:
                event_header = 'Event "%s"' % event
                event_condition = 'Event condition'
                event_code = 'Executed statement(s)'
            condition = self.events[event]
            text.append(r'<b>%s:</b><ul style="list-style-type: none; margin-top: 0px;">' % event_header)
            text.append(r'<li><i>%s: </i>' % event_condition)
            text.append('<code>%s</code></li>' % str(condition))
            statements = self.event_codes.get(event, None)
            if statements is not None:
                text.append(r'<li><i>%s:</i>' % event_code)
                if '\n' in str(statements):
                    text.append('</br>')
                text.append(r'<code>%s</code></li>' % str(statements))
            text.append('</ul>')

        if 'spike' in self.events:
            add_event_to_text('spike')
        for event in self.events:
            if event!='spike':
                add_event_to_text(event)

        return '\n'.join(text)
