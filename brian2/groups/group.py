'''
This module defines the `Group` object, a mix-in class for everything that
saves state variables, e.g. `NeuronGroup` or `StateMonitor`.
'''
import weakref
import copy
from collections import defaultdict

import numpy as np

from brian2.core.base import BrianObject
from brian2.core.variables import (ArrayVariable, StochasticVariable,
                                   AttributeVariable, AuxiliaryVariable,
                                   Variable)
from brian2.core.namespace import get_local_namespace
from brian2.units.fundamentalunits import (fail_for_dimension_mismatch, Unit)
from brian2.units.allunits import second
from brian2.codegen.translation import analyse_identifiers
from brian2.equations.unitcheck import check_units_statements
from brian2.utils.logger import get_logger
from brian2.devices.device import get_device


__all__ = ['Group', 'GroupCodeRunner']

logger = get_logger(__name__)


class Group(BrianObject):
    '''
    Mix-in class for accessing arrays by attribute.
    
    # TODO: Overwrite the __dir__ method to return the state variables
    # (should make autocompletion work)
    '''
    def _enable_group_attributes(self):
        if not hasattr(self, 'variables'):
            raise ValueError('Classes derived from Group need variables attribute.')
        if not hasattr(self, 'variable_indices'):
            self.variable_indices = defaultdict(lambda: '_idx')
        if not hasattr(self, 'codeobj_class'):
            self.codeobj_class = None
        self._group_attribute_access_active = True

    def _create_variables(self):
        '''
        Create standard set of variables every `Group` has, consisting of its
        clock's ``t`` and ``dt`` and the group's ``N``.
        '''
        return {'t': AttributeVariable(second, self.clock, 't_',
                                       constant=False, read_only=True),
                'dt': AttributeVariable(second, self.clock, 'dt_',
                                        constant=True, read_only=True),
                # This has to be overwritten for Synapses, since the number of
                # synapses is not known in the beginning
                'N': Variable(Unit(1), value=self._N, scalar=True,
                              constant=True, is_bool=False, read_only=True)
                }

    def state_(self, name):
        '''
        Gets the unitless array.
        '''
        try:
            var = self.variables[name]
        except KeyError:
            raise KeyError("State variable "+name+" not found.")

        return var.get_addressable_value(name=name, group=self)
        
    def state(self, name):
        '''
        Gets the array with units.
        '''
        try:
            var = self.variables[name]
        except KeyError:
            raise KeyError("State variable "+name+" not found.")

        return var.get_addressable_value_with_unit(name=name, group=self)

    def __getattr__(self, name):
        # We do this because __setattr__ and __getattr__ are not active until
        # _group_attribute_access_active attribute is set, and if it is set,
        # then __getattr__ will not be called. Therefore, if getattr is called
        # with this name, it is because it hasn't been set yet and so this
        # method should raise an AttributeError to agree that it hasn't been
        # called yet.
        if name=='_group_attribute_access_active':
            raise AttributeError
        if not hasattr(self, '_group_attribute_access_active'):
            raise AttributeError
        
        # We want to make sure that accessing variables without units is fast
        # because this is what is used during simulations
        # We do not specifically check for len(name) here, we simply assume
        # that __getattr__ is not called with an empty string (which wouldn't
        # be possibly using the normal dot syntax, anyway)
        try:
            if name[-1] == '_':
                origname = name[:-1]
                return self.state_(origname)
            else:
                return self.state(name)
        except KeyError:
            raise AttributeError('No attribute with name ' + name)

    def __setattr__(self, name, val):
        # attribute access is switched off until this attribute is created by
        # _enable_group_attributes
        if not hasattr(self, '_group_attribute_access_active') or name in self.__dict__:
            object.__setattr__(self, name, val)
        elif name in self.variables:
            var = self.variables[name]
            if not isinstance(val, basestring):
                fail_for_dimension_mismatch(val, var.unit,
                                            'Incorrect units for setting %s' % name)
            if var.read_only:
                raise TypeError('Variable %s is read-only.' % name)
            # Make the call X.var = ... equivalent to X.var[:] = ...
            var.get_addressable_value_with_unit(name, self, level=1)[slice(None)] = val
        elif len(name) and name[-1]=='_' and name[:-1] in self.variables:
            # no unit checking
            var = self.variables[name[:-1]]
            if var.read_only:
                raise TypeError('Variable %s is read-only.' % name[:-1])
            # Make the call X.var = ... equivalent to X.var[:] = ...
            var.get_addressable_value(name, self, level=1)[slice(None)] = val
        else:
            object.__setattr__(self, name, val)

    def calc_indices(self, item):
        '''
        Return flat indices from to index into state variables from arbitrary
        group specific indices. In the default implementation, raises an error
        for multidimensional indices and transforms slices into arrays.

        Parameters
        ----------
        item : slice, array, int
            The indices to translate.

        Returns
        -------
        indices : `numpy.ndarray`
            The flat indices corresponding to the indices given in `item`.
        '''
        if isinstance(item, tuple):
            raise IndexError(('Can only interpret 1-d indices, '
                              'got %d dimensions.') % len(item))
        else:
            if isinstance(item, slice):
                start, stop, step = item.indices(self.N)
                return np.arange(start, stop, step)
            else:
                index_array = np.asarray(item)
                if not np.issubdtype(index_array.dtype, np.int):
                    raise TypeError('Indexing is only supported for integer arrays')
                return index_array

    def _get_with_code(self, variable_name, variable, code, level=0):
        '''
        Gets a variable using a string expression. Is called by
        `VariableView.__getitem__` for statements such as
        ``print G.v['g_syn > 0']``

        Parameters
        ----------
        variable_name : str
            The name of the variable in its context (e.g. `'g_post'` for a
            variable with name `'g'`)
        variable : `ArrayVariable`
            The `ArrayVariable` object for the variable to be set
        code : str
            The code that should be executed to set the variable values.
            Can contain references to indices, such as `i` or `j`
        level : int, optional
            How much farther to go down in the stack to find the namespace.
            Necessary so that both `X.var = ` and `X.var[:] = ` have access
            to the surrounding namespace.
        '''
        # interpret the string expression
        namespace = get_local_namespace(level+1)
        additional_namespace = ('implicit-namespace', namespace)
        # Add the recorded variable under a known name to the variables
        # dictionary. Important to deal correctly with
        # the type of the variable in C++
        variables = {'_variable': AuxiliaryVariable(variable.unit,
                                                    dtype=variable.dtype,
                                                    scalar=variable.scalar,
                                                    is_bool=variable.is_bool),
                     '_cond': AuxiliaryVariable(Unit(1), dtype=np.bool,
                                                is_bool=True)}

        abstract_code = '_variable = ' + variable_name + '\n'
        abstract_code += '_cond = ' + code
        check_code_units(abstract_code, self,
                         additional_namespace=additional_namespace)
        codeobj = create_runner_codeobj(self,
                                        abstract_code,
                                        'state_variable_indexing',
                                        additional_variables=variables,
                                        additional_namespace=additional_namespace,
                                        )
        return codeobj()

    def _set_with_code(self, variable, group_indices, code, check_units=True,
                       level=0):
        '''
        Sets a variable using a string expression. Is called by
        `VariableView.__setitem__` for statements such as
        `S.var[:, :] = 'exp(-abs(i-j)/space_constant)*nS'`

        Parameters
        ----------
        variable : `ArrayVariable`
            The `ArrayVariable` object for the variable to be set
        group_indices : ndarray of int
            The indices of the elements that are to be set.
        code : str
            The code that should be executed to set the variable values.
            Can contain references to indices, such as `i` or `j`
        check_units : bool, optional
            Whether to check the units of the expression.
        level : int, optional
            How much farther to go down in the stack to find the namespace.
            Necessary so that both `X.var = ` and `X.var[:] = ` have access
            to the surrounding namespace.
        '''
        abstract_code = variable.name + ' = ' + code
        namespace = get_local_namespace(level + 1)
        additional_namespace = ('implicit-namespace', namespace)
        additional_variables = {'_group_idx': ArrayVariable('_group_idx',
                                                            Unit(1),
                                                            value=group_indices.astype(np.int32),
                                                            group_name=self.name)}
        # TODO: Have an additional argument to avoid going through the index
        # array for situations where iterate_all could be used
        codeobj = create_runner_codeobj(self,
                                 abstract_code,
                                 'group_variable_set',
                                 additional_variables=additional_variables,
                                 additional_namespace=additional_namespace,
                                 check_units=check_units)
        codeobj()

    def _set_with_code_conditional(self, variable, cond, code, check_units=True,
                                   level=0):
        '''
        Sets a variable using a string expression and string condition. Is
        called by `VariableView.__setitem__` for statements such as
        `S.var['i!=j'] = 'exp(-abs(i-j)/space_constant)*nS'`

        Parameters
        ----------
        variable : `ArrayVariable`
            The `ArrayVariable` object for the variable to be set.
        cond : str
            The string condition for which the variables should be set.
        code : str
            The code that should be executed to set the variable values.
        check_units : bool, optional
            Whether to check the units of the expression.
        level : int, optional
            How much farther to go down in the stack to find the namespace.
            Necessary so that both `X.var = ` and `X.var[:] = ` have access
            to the surrounding namespace.
        '''

        abstract_code_cond = '_cond = '+cond
        abstract_code = variable.name + ' = ' + code
        namespace = get_local_namespace(level + 1)
        additional_namespace = ('implicit-namespace', namespace)
        check_code_units(abstract_code_cond, self,
                         additional_namespace=additional_namespace)
        # TODO: Have an additional argument to avoid going through the index
        # array for situations where iterate_all could be used
        codeobj = create_runner_codeobj(self,
                                 {'condition': abstract_code_cond, 'statement': abstract_code},
                                 'group_variable_set_conditional',
                                 additional_namespace=additional_namespace,
                                 check_units=check_units)
        codeobj()


def check_code_units(code, group, additional_variables=None,
                     additional_namespace=None,
                     ignore_keyerrors=False):
    '''
    Check statements for correct units.

    Parameters
    ----------
    code : str
        The series of statements to check
    group : `Group`
        The context for the code execution
    additional_variables : dict-like, optional
        A mapping of names to `Variable` objects, used in addition to the
        variables saved in `self.group`.
    additional_namespace : dict-like, optional
        An additional namespace, as provided to `Group.before_run`
    ignore_keyerrors : boolean, optional
        Whether to silently ignore unresolvable identifiers. Should be set
        to ``False`` (the default) if the namespace is expected to be
        complete (e.g. in `Group.before_run`) but to ``True`` when the check
        is done during object initialisation where the namespace is not
        necessarily complete yet.

    Raises
    ------
    DimensionMismatchError
        If `code` has unit mismatches
    '''
    all_variables = dict(group.variables)
    if additional_variables is not None:
        all_variables.update(additional_variables)

    # Resolve the namespace, resulting in a dictionary containing only the
    # external variables that are needed by the code -- keep the units for
    # the unit checks
    # Note that here we do not need to recursively descend into
    # subexpressions. For unit checking, we only need to know the units of
    # the subexpressions not what variables they refer to
    _, _, unknown = analyse_identifiers(code, all_variables)
    try:
        resolved_namespace = group.namespace.resolve_all(unknown,
                                                         additional_namespace,
                                                         strip_units=False)
    except KeyError as ex:
        if ignore_keyerrors:
            logger.debug('Namespace not complete (yet), ignoring: %s ' % str(ex),
                         'check_code_units')
            return
        else:
            raise KeyError('Error occured when checking "%s": %s' % (code,
                                                                     str(ex)))

    check_units_statements(code, resolved_namespace, all_variables)


def create_runner_codeobj(group, code, template_name,
                          variable_indices=None,
                          name=None, check_units=True,
                          needed_variables=None,
                          additional_variables=None,
                          additional_namespace=None,
                          template_kwds=None):
    ''' Create a `CodeObject` for the execution of code in the context of a
    `Group`.

    Parameters
    ----------
    group : `Group`
        The group where the code is to be run
    code : str
        The code to be executed.
    template_name : str
        The name of the template to use for the code.
    variable_indices : dict-like, optional
        A mapping from `Variable` objects to index names (strings).  If none is
        given, uses the corresponding attribute of `group`.
    name : str, optional
        A name for this code object, will use ``group + '_codeobject*'`` if
        none is given.
    check_units : bool, optional
        Whether to check units in the statement. Defaults to ``True``.
    needed_variables: list of str, optional
        A list of variables that are neither present in the abstract code, nor
        in the ``USES_VARIABLES`` statement in the template. This is only
        rarely necessary, an example being a `StateMonitor` where the
        names of the variables are neither known to the template nor included
        in the abstract code statements.
    additional_variables : dict-like, optional
        A mapping of names to `Variable` objects, used in addition to the
        variables saved in `group`.
    additional_namespace : dict-like, optional
        A mapping from names to objects, used in addition to the namespace
        saved in `group`.
    template_kwds : dict, optional
        A dictionary of additional information that is passed to the template.
    '''
    logger.debug('Creating code object for abstract code:\n' + str(code))

    if check_units:
        if isinstance(code, dict):
            for c in code.values():
                check_code_units(c, group,
                                 additional_variables=additional_variables,
                                 additional_namespace=additional_namespace)
        else:
            check_code_units(code, group,
                             additional_variables=additional_variables,
                             additional_namespace=additional_namespace)

    codeobj_class = get_device().code_object_class(group.codeobj_class)
    template = getattr(codeobj_class.templater, template_name)

    all_variables = dict(group.variables)
    if additional_variables is not None:
        all_variables.update(additional_variables)
        
    # Determine the identifiers that were used
    if isinstance(code, dict):
        used_known = set()
        unknown = set()
        for v in code.values():
            _, uk, u = analyse_identifiers(v, all_variables, recursive=True)
            used_known |= uk
            unknown |= u
    else:
        _, used_known, unknown = analyse_identifiers(code, all_variables,
                                                     recursive=True)

    logger.debug('Unknown identifiers in the abstract code: ' + str(unknown))

    # Only pass the variables that are actually used
    variables = {}
    for var in used_known:
        # Emit a warning if a variable is also present in the namespace
        if var in group.namespace or var in additional_namespace[1]:
            message = ('Variable {var} is present in the namespace but is also an'
                       ' internal variable of {name}, the internal variable will'
                       ' be used.'.format(var=var, name=group.name))
            logger.warn(message, 'create_runner_codeobj.resolution_conflict',
                        once=True)
        if not isinstance(all_variables[var], StochasticVariable):
            variables[var] = all_variables[var]

    resolved_namespace = group.namespace.resolve_all(unknown,
                                                     additional_namespace)

    # Add variables that are not in the abstract code, nor specified in the
    # template but nevertheless necessary
    if needed_variables is None:
        needed_variables = []
    for var in needed_variables:
        variables[var] = all_variables[var]

    # Also add the variables that the template needs
    for var in template.variables:
        try:
            variables[var] = all_variables[var]
        except KeyError as ex:
            # We abuse template.variables here to also store names of things
            # from the namespace (e.g. rand) that are needed
            # TODO: Improve all of this namespace/specifier handling
            if group is not None:
                # Try to find the name in the group's namespace
                resolved_namespace[var] = group.namespace.resolve(var,
                                                                  additional_namespace)
            else:
                raise ex

    # always add N, the number of neurons or synapses
    variables['N'] = all_variables['N']

    if name is None:
        if group is not None:
            name = '%s_%s_codeobject*' % (group.name, template_name) 
        else:
            name = '%s_codeobject*' % template_name

    all_variable_indices = copy.copy(group.variable_indices)
    if variable_indices is not None:
        all_variable_indices.update(variable_indices)

    # Add the indices needed by the variables
    varnames = variables.keys()
    for varname in varnames:
        var_index = all_variable_indices[varname]
        if var_index != '_idx':
            variables[var_index] = all_variables[var_index]

    return get_device().code_object(
                             group,
                             name,
                             code,
                             resolved_namespace,
                             variables,
                             template_name,
                             variable_indices=all_variable_indices,
                             template_kwds=template_kwds,
                             codeobj_class=group.codeobj_class)


class GroupCodeRunner(BrianObject):
    '''
    A "runner" that runs a `CodeObject` every timestep and keeps a reference to
    the `Group`. Used in `NeuronGroup` for `Thresholder`, `Resetter` and
    `StateUpdater`.
    
    On creation, we try to run the before_run method with an empty additional
    namespace (see `Network.before_run`). If the namespace is already complete
    this might catch unit mismatches.
    
    Parameters
    ----------
    group : `Group`
        The group to which this object belongs.
    template : `Template`
        The template that should be used for code generation
    code : str, optional
        The abstract code that should be executed every time step. The
        `update_abstract_code` method might generate this code dynamically
        before every run instead.
    when : `Scheduler`, optional
        At which point in the schedule this object should be executed.
    name : str, optional 
        The name for this object.
    check_units : bool, optional
        Whether the units should be checked for consistency before a run. Is
        activated (``True``) by default but should be switched off for state
        updaters (units are already checked for the equations and the generated
        abstract code might have already replaced variables with their unit-less
        values)
    template_kwds : dict, optional
        A dictionary of additional information that is passed to the template.
    needed_variables: list of str, optional
        A list of variables that are neither present in the abstract code, nor
        in the ``USES_VARIABLES`` statement in the template. This is only
        rarely necessary, an example being a `StateMonitor` where the
        names of the variables are neither known to the template nor included
        in the abstract code statements.
    '''
    def __init__(self, group, template, code=None, when=None,
                 name='coderunner*', check_units=True, template_kwds=None,
                 needed_variables=None):
        BrianObject.__init__(self, when=when, name=name)
        self.group = weakref.proxy(group)
        self.template = template
        self.abstract_code = code
        self.check_units = check_units
        if needed_variables is None:
            needed_variables = []
        self.needed_variables = needed_variables
        self.template_kwds = template_kwds
    
    def update_abstract_code(self):
        '''
        Update the abstract code for the code object. Will be called in
        `before_run` and should update the `GroupCodeRunner.abstract_code`
        attribute.
        
        Does nothing by default.
        '''
        pass

    def before_run(self, namespace):
        self.update_abstract_code()
        # If the GroupCodeRunner has variables, add them
        if hasattr(self, 'variables'):
            additional_variables = self.variables
        else:
            additional_variables = None
        if self.check_units:
            check_code_units(self.abstract_code, self.group,
                             additional_variables, namespace)
        self.codeobj = create_runner_codeobj(self.group, self.abstract_code,
                                             self.template,
                                             name=self.name+'_codeobject*',
                                             check_units=self.check_units,
                                             additional_variables=additional_variables,
                                             additional_namespace=namespace,
                                             needed_variables=self.needed_variables,
                                             template_kwds=self.template_kwds)
        self.code_objects[:] = [weakref.proxy(self.codeobj)]
        self.updaters[:] = [self.codeobj.get_updater()]
