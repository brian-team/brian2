'''
This module defines the `Group` object, a mix-in class for everything that
saves state variables, e.g. `NeuronGroup` or `StateMonitor`.
'''
import weakref
from collections import defaultdict

import numpy as np

from brian2.core.base import BrianObject
from brian2.core.variables import (ArrayVariable, StochasticVariable,
                                   AttributeVariable, Variable)
from brian2.core.namespace import get_local_namespace
from brian2.units.fundamentalunits import fail_for_dimension_mismatch, Unit
from brian2.units.allunits import second
from brian2.codegen.codeobject import get_codeobject_template, create_codeobject
from brian2.codegen.translation import analyse_identifiers
from brian2.equations.unitcheck import check_units_statements
from brian2.utils.logger import get_logger
from brian2.devices.device import get_device

__all__ = ['Group', 'GroupCodeRunner']

logger = get_logger(__name__)


class GroupItemMapping(Variable):

    def __init__(self, N, offset, group):
        self.N = N
        self.offset = int(offset)
        self.group = weakref.proxy(group)
        self._indices = np.arange(self.N + self.offset)
        self.variables = {'i': ArrayVariable('i',
                                              Unit(1),
                                              self._indices - self.offset)}
        Variable.__init__(self, Unit(1), value=self, constant=True)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        '''
        Returns indices for `index` an array, integer or slice, or a string
        (that might refer to ``i`` as the group element index).

        '''
        if isinstance(index, tuple):
            raise IndexError(('Can only interpret 1-d indices, '
                              'got %d dimensions.') % len(index))
        if isinstance(index, basestring):
            # interpret the string expression
            namespace = get_local_namespace(1)
            additional_namespace = ('implicit-namespace', namespace)
            abstract_code = '_cond = ' + index
            check_code_units(abstract_code, self.group,
                             additional_variables=self.variables,
                             additional_namespace=additional_namespace)
            template = getattr(self.group, '_index_with_code_template',
                              'state_variable_indexing')
            codeobj = create_runner_codeobj(self.group,
                                            abstract_code,
                                            template,
                                            additional_variables=self.variables,
                                            additional_namespace=additional_namespace,
                                            )
            return codeobj()
        else:
            if isinstance(index, slice):
                start, stop, step = index.indices(self.N)
                index = slice(start + self.offset, stop + self.offset, step)
                return self._indices[index]
            else:
                index_array = np.asarray(index)
                if not np.issubdtype(index_array.dtype, np.int):
                    raise TypeError('Indexing is only supported for integer arrays')
                return self._indices[index_array + self.offset]


class Group(object):
    '''
    Mix-in class for accessing arrays by attribute.
    
    # TODO: Overwrite the __dir__ method to return the state variables
    # (should make autocompletion work)
    '''
    def __init__(self):
        if not hasattr(self, 'offset'):
            self.offset = 0
        if not hasattr(self, 'variables'):
            raise ValueError('Classes derived from Group need variables attribute.')
        if not hasattr(self, 'item_mapping'):
            try:
                N = len(self)
            except TypeError:
                raise ValueError(('Classes derived from Group need an item_mapping '
                                  'attribute, or a length to automatically '
                                  'provide 1-d indexing'))
            self.item_mapping = GroupItemMapping(N, self.offset, self)
        if not hasattr(self, 'indices'):
            self.indices = {'_idx': self.item_mapping}
        if not hasattr(self, 'variable_indices'):
            self.variable_indices = defaultdict(lambda: '_idx')
        if not hasattr(self, 'codeobj_class'):
            self.codeobj_class = None
        self._group_attribute_access_active = True

    def _create_variables(self):
        return {'t': AttributeVariable(second, self.clock, 't_',
                                       constant=False),
                'dt': AttributeVariable(second, self.clock, 'dt_',
                                        constant=True)
                }

    def state_(self, name):
        '''
        Gets the unitless array.
        '''
        try:
            return self.variables[name].get_addressable_value(self)
        except KeyError:
            raise KeyError("Array named "+name+" not found.")
        
    def state(self, name):
        '''
        Gets the array with units.
        '''
        try:
            var = self.variables[name]
            return var.get_addressable_value_with_unit(self)
        except KeyError:
            raise KeyError("Array named "+name+" not found.")

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
        # Group.__init__
        if not hasattr(self, '_group_attribute_access_active'):
            object.__setattr__(self, name, val)
        elif name in self.variables:
            var = self.variables[name]
            if not isinstance(val, basestring):
                fail_for_dimension_mismatch(val, var.unit,
                                            'Incorrect units for setting %s' % name)
            # Make the call X.var = ... equivalent to X.var[:] = ...
            var.get_addressable_value_with_unit(self, level=1)[:] = val
        elif len(name) and name[-1]=='_' and name[:-1] in self.variables:
            # no unit checking
            var = self.variables[name[:-1]]
            # Make the call X.var = ... equivalent to X.var[:] = ...
            var.get_addressable_value(self, level=1)[:] = val
        else:
            object.__setattr__(self, name, val)

    def _set_with_code(self, variable, group_indices, code,
                       template, check_units=True, level=0):
        '''
        Sets a variable using a string expression. Is called by
        `VariableView.__setitem__` for statements such as
        `S.var[:, :] = 'exp(-abs(i-j)/space_constant)*nS'`

        Parameters
        ----------
        variable : `ArrayVariable`
            The `Variable` for the variable to be set
        group_indices : ndarray of int
            The indices of the elements that are to be set.
        code : str
            The code that should be executed to set the variable values.
            Can contain references to indices, such as `i` or `j`
        template : str
            The name of the template to use.
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
        additional_variables = self.item_mapping.variables
        additional_variables['_group_idx'] = ArrayVariable('_group_idx',
                                                     Unit(1),
                                                     value=group_indices.astype(np.int32),
                                                     group_name=self.name)
        # TODO: Have an additional argument to avoid going through the index
        # array for situations where iterate_all could be used
        codeobj = create_runner_codeobj(self,
                                 abstract_code,
                                 template,
                                 additional_variables=additional_variables,
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
        An additional namespace, as provided to `Group.pre_run`
    ignore_keyerrors : boolean, optional
        Whether to silently ignore unresolvable identifiers. Should be set
         to ``False`` (the default) if the namespace is expected to be
         complete (e.g. in `Group.pre_run`) but to ``True`` when the check
         is done during object initialisation where the namespace is not
         necessarily complete yet

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
            raise ex

    check_units_statements(code, resolved_namespace, all_variables)


def create_runner_codeobj(group, code, template_name, indices=None,
                          variable_indices=None,
                          name=None, check_units=True,
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
    template : `LanguageTemplater`
        The template to use for the code.
    indices : dict-like, optional
        A mapping from index name to `Index` objects, describing the indices
        used for the variables in the code. If none are given, uses the
        corresponding attribute of `group`.
    variable_indices : dict-like, optional
        A mapping from `Variable` objects to index names (strings).  If none is
        given, uses the corresponding attribute of `group`.
    name : str, optional
        A name for this code object, will use ``group + '_codeobject*'`` if
        none is given.
    check_units : bool, optional
        Whether to check units in the statement. Defaults to ``True``.
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
        check_code_units(code, group, additional_variables=additional_variables,
                         additional_namespace=additional_namespace)

    template = get_codeobject_template(template_name,
                                       codeobj_class=group.codeobj_class)

    all_variables = dict(group.variables)
    if additional_variables is not None:
        all_variables.update(additional_variables)

    # Determine the identifiers that were used
    _, used_known, unknown = analyse_identifiers(code, all_variables,
                                                 recursive=True)

    logger.debug('Unknown identifiers in the abstract code: ' + str(unknown))

    # Only pass the variables that are actually used
    variables = {}
    for var in used_known:
        if not isinstance(all_variables[var], StochasticVariable):
            variables[var] = all_variables[var]

    resolved_namespace = group.namespace.resolve_all(unknown,
                                                     additional_namespace)

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

    if name is None:
        if group is not None:
            name = '%s_%s_codeobject*' % (group.name, template_name) 
        else:
            name = '%s_codeobject*' % template_name

    if indices is None:
        indices = group.indices
    if variable_indices is None:
        variable_indices = group.variable_indices

    return get_device().code_object(
                             name,
                             code,
                             resolved_namespace,
                             variables,
                             template_name,
                             indices=indices,
                             variable_indices=variable_indices,
                             template_kwds=template_kwds,
                             codeobj_class=group.codeobj_class)


class GroupCodeRunner(BrianObject):
    '''
    A "runner" that runs a `CodeObject` every timestep and keeps a reference to
    the `Group`. Used in `NeuronGroup` for `Thresholder`, `Resetter` and
    `StateUpdater`.
    
    On creation, we try to run the pre_run method with an empty additional
    namespace (see `Network.pre_run`). If the namespace is already complete
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
    
    Notes
    -----
    Objects such as `Thresholder`, `Resetter` or `StateUpdater` inherit from
    this class. They can customize the behaviour by overwriting the
    `update_abstract_code`, `pre_update` and `post_update` method.
    `update_abstract_code` is called before a run to allow taking into account
    changes in the namespace or in the reset/threshold definition itself.
    `pre_update` and `post_update` are used to connect the `CodeObject` to the
    state of the `Group`. For example, the `Thresholder` sets the
    `NeuronGroup.spikes` property in `post_update`.
    '''
    def __init__(self, group, template, code=None, when=None,
                 name='coderunner*', check_units=True, template_kwds=None):
        BrianObject.__init__(self, when=when, name=name)
        self.group = weakref.proxy(group)
        self.template = template
        self.abstract_code = code
        self.check_units = check_units
        self.template_kwds = template_kwds
    
    def update_abstract_code(self):
        '''
        Update the abstract code for the code object. Will be called in
        `pre_run` and should update the `GroupCodeRunner.abstract_code`
        attribute.
        
        Does nothing by default.
        '''
        pass

    def pre_run(self, namespace):
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
                                             template_kwds=self.template_kwds)
        self.code_objects[:] = [weakref.proxy(self.codeobj)]
    
    def pre_update(self):
        '''
        Will be called in every timestep before the `update` method is called.
        
        Does nothing by default.
        '''
        pass
    
    def update(self, **kwds):
        self.pre_update()
        return_value = self.codeobj(**kwds)
        self.post_update(return_value)

    def post_update(self, return_value):
        '''
        Will be called in every timestep after the `update` method is called.
        
        Overwritten in `Thresholder` to update the ``spikes`` list saved in 
        a `NeuronGroup`.
        
        Does nothing by default.
        
        Parameters
        ----------
        return_value : object
            The result returned from calling the `CodeObject`.
        '''
        pass
