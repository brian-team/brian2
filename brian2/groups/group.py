'''
This module defines the `Group` object, a mix-in class for everything that
saves state variables, e.g. `NeuronGroup` or `StateMonitor`.
'''
import collections
import weakref
try:
    from collections import OrderedDict
except ImportError:
    # OrderedDict was added in Python 2.7, use backport for Python 2.6
    from brian2.utils.ordereddict import OrderedDict

import numpy as np

from brian2.core.base import BrianObject
from brian2.core.preferences import brian_prefs
from brian2.core.variables import Variables, Constant, Variable, Subexpression
from brian2.core.functions import Function
from brian2.core.namespace import (get_local_namespace,
                                   DEFAULT_FUNCTIONS,
                                   DEFAULT_UNITS,
                                   DEFAULT_CONSTANTS)
from brian2.core.scheduler import Scheduler
from brian2.codegen.codeobject import create_runner_codeobj, check_code_units
from brian2.equations.equations import BOOLEAN, INTEGER, FLOAT
from brian2.units.fundamentalunits import (fail_for_dimension_mismatch, Unit,
                                           get_unit)
from brian2.utils.logger import get_logger
from brian2.utils.stringtools import get_identifiers

__all__ = ['Group', 'CodeRunner']

logger = get_logger(__name__)


def _conflict_warning(message, resolutions):
    '''
    A little helper functions to generate warnings for logging. Specific
    to the `Namespace.resolve` method and should only be used by it.

    Parameters
    ----------
    message : str
        The first part of the warning message.
    resolutions : list of str
        A list of (namespace, object) tuples.
    '''
    if len(resolutions) == 0:
        # nothing to warn about
        return
    elif len(resolutions) == 1:
        second_part = ('but also refers to a variable in the %s namespace:'
                       ' %r') % (resolutions[0][0], resolutions[0][1])
    else:
        second_part = ('but also refers to a variable in the following '
                       'namespaces: %s') % (', '.join([r[0] for r in resolutions]))

    logger.warn(message + ' ' + second_part,
                'Group.resolve.resolution_conflict', once=True)


def get_dtype(equation, dtype=None):
    '''
    Helper function to interpret the `dtype` keyword argument in `NeuronGroup`
    etc.

    Parameters
    ----------
    equation : `SingleEquation`
        The equation for which a dtype should be returned
    dtype : `dtype` or dict, optional
        Either the `dtype` to be used as a default dtype for all float variables
        (instead of the `core.default_float_dtype` preference) or a
        dictionary stating the `dtype` for some variables; all other variables
        will use the preference default

    Returns
    -------
    d : `dtype`
        The dtype for the variable defined in `equation`
    '''
    # Check explicitly provided dtype for compatibility with the variable type
    if isinstance(dtype, collections.Mapping):
        if equation.varname in dtype:
            BASIC_TYPES = {BOOLEAN: 'b',
                           INTEGER: 'iu',
                           FLOAT: 'f'}
            provided_dtype = np.dtype(dtype[equation.varname])
            if not provided_dtype.kind in BASIC_TYPES[equation.var_type]:
                raise TypeError(('Error determining dtype for variable %s: %s '
                                 'is not a correct type for %s variables') % (equation.varname,
                                                              provided_dtype.name,
                                                              equation.var_type))
            else:
                return dtype[equation.varname]
        else:  # continue as if no dtype had been specified at all
            dtype = None

    # Use default dtypes (or a provided standard dtype for floats)
    if equation.var_type == BOOLEAN:
        return np.bool
    elif equation.var_type == INTEGER:
        return brian_prefs['core.default_integer_dtype']
    elif equation.var_type == FLOAT:
        if dtype is not None:
            dtype = np.dtype(dtype)
            if not dtype.kind == 'f':
                raise TypeError(('%s is not a valid floating point '
                                 'dtype') % dtype)
            return dtype
        else:
            return brian_prefs['core.default_float_dtype']
    else:
        raise ValueError(('Do not know how to determine a dtype for '
                          'variable %s of type %s' ) % (equation.varname,
                                                        equation.var_type))


def _same_value(obj1, obj2):
    '''
    Helper function used during namespace resolution.
    '''
    if obj1 is obj2:
        return True
    try:
        obj1 = obj1.get_value()
    except (AttributeError, TypeError):
        pass

    try:
        obj2 = obj2.get_value()
    except (AttributeError, TypeError):
        pass

    return obj1 is obj2


def _same_function(func1, func2):
    '''
    Helper function, used during namespace resolution for comparing whether to
    functions are the same. This takes care of treating a function and a
    `Function` variables whose `Function.pyfunc` attribute matches as the
    same. This prevents the user from getting spurious warnings when having
    for example a numpy function such as :np:func:`~random.randn` in the local
    namespace, while the ``randn`` symbol in the numpy namespace used for the
    code objects refers to a `RandnFunction` specifier.
    '''
    # use the function itself if it doesn't have a pyfunc attribute
    func1 = getattr(func1, 'pyfunc', func1)
    func2 = getattr(func2, 'pyfunc', func2)

    return func1 is func2


class Indexing(object):
    '''
    Object responsible for calculating flat index arrays from arbitrary group-
    specific indices. Stores strong references to the necessary variables so
    that basic indexing (i.e. slicing, integer arrays/values, ...) works even
    when the respective `Group` no longer exists. Note that this object does
    not handle string indexing.
    '''
    def __init__(self, group):
        self.group = weakref.proxy(group)
        self.N = group.variables['N']
        self.index_variables = dict([(varname, group.variables[varname])
                                     for varname in set(group.variables.indices.values())
                                     if not varname in ('_idx', '0')])

    def calc_indices(self, item, var_index='_idx'):
        '''
        Return flat indices to index into state variables from arbitrary
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

        See Also
        --------
        SynapticIndexing
        '''
        if isinstance(item, tuple):
            raise IndexError(('Can only interpret 1-d indices, '
                              'got %d dimensions.') % len(item))
        else:
            if isinstance(item, slice):
                if var_index == '_idx':
                    start, stop, step = item.indices(self.N.get_value())
                else:
                    # For linked variables, the index might not have been there
                    # yet at the time of the creation of the indexing object
                    if var_index in self.index_variables:
                        start, stop, step = item.indices(self.index_variables[var_index].size)
                    else:
                        start, stop, step = item.indices(self.group.variables[var_index].size)
                return np.arange(start, stop, step)
            else:
                index_array = np.asarray(item)
                if index_array.dtype == np.bool:
                    index_array = np.nonzero(index_array)[0]
                elif not np.issubdtype(index_array.dtype, np.int):
                    raise TypeError(('Indexing is only supported for integer '
                                     'and boolean arrays, not for type '
                                     '%s' % index_array.dtype))
                return index_array


class IndexWrapper(object):
    '''
    Convenience class to allow access to the indices via indexing syntax. This
    allows for example to get all indices for synapses originating from neuron
    10 by writing `synapses.indices[10, :]` instead of
    `synapses._indexing.calc_indices((10, slice(None))`.
    '''
    def __init__(self, group):
        self.group = weakref.proxy(group)
        self.indices = group._indexing

    def __getitem__(self, item):
        if isinstance(item, basestring):
            variables = Variables(None)
            variables.add_auxiliary_variable('_indices', unit=Unit(1),
                                             dtype=np.int32)
            variables.add_auxiliary_variable('_cond', unit=Unit(1),
                                             dtype=np.bool)

            abstract_code = '_cond = ' + item
            check_code_units(abstract_code, self.group,
                             additional_variables=variables,
                             level=1)
            codeobj = create_runner_codeobj(self.group,
                                            abstract_code,
                                            'group_get_indices',
                                            additional_variables=variables,
                                            level=1
                                            )
            return codeobj()
        else:
            return self.indices.calc_indices(item)


class Group(BrianObject):
    '''
    Mix-in class for accessing arrays by attribute.
    
    # TODO: Overwrite the __dir__ method to return the state variables
    # (should make autocompletion work)
    '''
    #: The class to convert group-specific indexing into 1d indices
    indexing_class = Indexing

    def _enable_group_attributes(self):
        if not hasattr(self, 'variables'):
            raise ValueError('Classes derived from Group need variables attribute.')
        if not hasattr(self, 'codeobj_class'):
            self.codeobj_class = None
        if not hasattr(self, 'indexing'):
            self._indexing = self.indexing_class(self)
        if not hasattr(self, 'indices'):
            self.indices = IndexWrapper(self)

        self._group_attribute_access_active = True

    def state(self, name, use_units=True, level=0):
        '''
        Return the state variable in a way that properly supports indexing in
        the context of this group

        Parameters
        ----------
        name : str
            The name of the state variable
        use_units : bool, optional
            Whether to use the state variable's unit.
        level : int, optional
            How much farther to go down in the stack to find the namespace.
        Returns
        -------
        var : `VariableView` or scalar value
            The state variable's value that can be indexed (for non-scalar
            values).
        '''
        try:
            var = self.variables[name]
        except KeyError:
            raise KeyError("State variable "+name+" not found.")

        if use_units:
            return var.get_addressable_value_with_unit(name=name, group=self)
        else:
            return var.get_addressable_value(name=name, group=self)

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
        # be possible using the normal dot syntax, anyway)
        try:
            if name[-1] == '_':
                name = name[:-1]
                use_units = False
            else:
                use_units = True
            return self.state(name, use_units)

        except KeyError:
            raise AttributeError('No attribute with name ' + name)

    def __setattr__(self, name, val, level=0):
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
            var.get_addressable_value_with_unit(name, self).set_item(slice(None),
                                                                     val,
                                                                     level=level+1)
        elif len(name) and name[-1]=='_' and name[:-1] in self.variables:
            # no unit checking
            var = self.variables[name[:-1]]
            if var.read_only:
                raise TypeError('Variable %s is read-only.' % name[:-1])
            # Make the call X.var = ... equivalent to X.var[:] = ...
            var.get_addressable_value(name[:-1], self).set_item(slice(None),
                                                                val,
                                                                level=level+1)
        else:
            object.__setattr__(self, name, val)

    def _check_expression_scalar(self, expr, varname, level=0,
                                 run_namespace=None):
        '''
        Helper function to check that an expression only refers to scalar
        variables, used when setting a scalar variable with a string expression.

        Parameters
        ----------
        expr : str
            The expression to check.
        varname : str
            The variable that is being set (only used for the error message)
        level : int, optional
            How far to go up in the stack to find the local namespace (if
            `run_namespace` is not set).
        run_namespace : dict-like, optional
            A specific namespace provided for this expression.

        Raises
        ------
        ValueError
            If the expression refers to a non-scalar variable.
        '''
        identifiers = get_identifiers(expr)
        referred_variables = self.resolve_all(identifiers,
                                              run_namespace=run_namespace,
                                              level=level+1)
        for ref_varname, ref_var in referred_variables.iteritems():
            if not getattr(ref_var, 'scalar', False):
                raise ValueError(('String expression for setting scalar '
                                  'variable %s refers to %s which is not '
                                  'scalar.') % (varname, ref_varname))

    def resolve(self, identifier, additional_variables=None,
                run_namespace=None, level=0, do_warn=True):
        '''
        Resolve an identifier (i.e. variable, constant or function name) in the
        context of this group. This function will first lookup the name in the
        state variables, then look for a standard function or unit of that
        name and finally look in `Group.namespace` and in `run_namespace`. If
        the latter is not given, it will try to find the variable in the local
        namespace where the original function call took place. See
        :ref:`external-variables`.

        Parameters
        ----------
        identifiers : str
            The name to look up.
        additional_variables : dict-like, optional
            An additional mapping of names to `Variable` objects that will be
            checked before `Group.variables`.
        run_namespace : dict-like, optional
            An additional namespace, provided as an argument to the
            `Network.run` method.
        level : int, optional
            How far to go up in the stack to find the original call frame.
        do_warn : bool, optional
            Whether to warn about names that are defined both as an internal
            variable (i.e. in `Group.variables`) and in some other namespace.
            Defaults to ``True`` but can be switched off for internal variables
            used in templates that the user might not even know about.

        Returns
        -------
        obj : `Variable` or `Function`
            Returns a `Variable` object describing the variable or a `Function`
            object for a function. External variables are represented as
            `Constant` objects

        Raises
        ------
        KeyError
            If the `identifier` could not be resolved
        '''
        resolved_internal = None

        if identifier in (additional_variables or {}):
            resolved_internal = additional_variables[identifier]
        elif identifier in getattr(self, 'variables', {}):
            resolved_internal = self.variables[identifier]

        if resolved_internal is not None:
            if do_warn is False:
                return resolved_internal  # no need to go further
            # We already found the identifier, but we try to resolve it in the
            # external namespace nevertheless, to report a warning if it is
            # present there as well.
            try:
                self._resolve_external(identifier,
                                       run_namespace=run_namespace,
                                       level=level+1,
                                       do_warn=False)
                # If we arrive here without a KeyError then the name is present
                # in the external namespace as well
                message = ('Variable {var} is present in the namespace but is '
                           'also an internal variable of {name}, the internal '
                           'variable will be used.'.format(var=identifier,
                                                           name=self.name))
                logger.warn(message, 'Group.resolve.resolution_conflict',
                            once=True)
            except KeyError:
                pass  # Nothing to warn about

            return resolved_internal

        # We did not find the name internally, try to resolve it in the external
        # namespace
        return self._resolve_external(identifier, run_namespace=run_namespace,
                                      level=level+1)

    def resolve_all(self, identifiers, additional_variables=None,
                    run_namespace=None, level=0, do_warn=True):
        '''
        Resolve a list of identifiers. Calls `Group.resolve` for each
        identifier.

        Parameters
        ----------
        identifiers : iterable of str
            The names to look up.
        additional_variables : dict-like, optional
            An additional mapping of names to `Variable` objects that will be
            checked before `Group.variables`.
        run_namespace : dict-like, optional
            An additional namespace, provided as an argument to the
            `Network.run` method.
        level : int, optional
            How far to go up in the stack to find the original call frame.
        do_warn : bool, optional
            Whether to warn about names that are defined both as an internal
            variable (i.e. in `Group.variables`) and in some other namespace.
            Defaults to ``True`` but can be switched off for internal variables
            used in templates that the user might not even know about.

        Returns
        -------
        variables : dict of `Variable` or `Function`
            A mapping from name to `Variable`/`Function` object for each of the
            names given in `identifiers`

        Raises
        ------
        KeyError
            If one of the names in `identifier` cannot be resolved
        '''
        resolved = {}
        for identifier in identifiers:
            resolved[identifier] = self.resolve(identifier,
                                                additional_variables=additional_variables,
                                                run_namespace=run_namespace,
                                                level=level+1,
                                                do_warn=do_warn)
        return resolved

    def _resolve_external(self, identifier, run_namespace=None, level=0,
                          do_warn=True):
        '''
        Resolve an external identifier in the context of a `Group`. If the `Group`
        declares an explicit namespace, this namespace is used in addition to the
        standard namespace for units and functions. Additionally, the namespace in
        the `run_namespace` argument (i.e. the namespace provided to `Network.run`)
        or, if this argument is unspecified, the implicit namespace of
        surrounding variables in the stack frame where the original call was made
        is used (to determine this stack frame, the `level` argument has to be set
        correctly).

        Parameters
        ----------
        identifier : str
            The name to resolve.
        group : `Group`
            The group that potentially defines an explicit namespace for looking up
            external names.
        run_namespace : dict, optional
            A namespace (mapping from strings to objects), as provided as an
            argument to the `Network.run` function.
        level : int, optional
            How far to go up in the stack to find the calling frame.
        do_warn : int, optional
            Whether to display a warning if an identifier resolves to different
            objects in different namespaces. Defaults to ``True``.
        '''
        # We save tuples of (namespace description, referred object) to
        # give meaningful warnings in case of duplicate definitions
        matches = []

        namespaces = OrderedDict()
        # Default namespaces (units and functions)
        namespaces['constants'] = DEFAULT_CONSTANTS
        namespaces['units'] = DEFAULT_UNITS
        namespaces['functions'] = DEFAULT_FUNCTIONS
        if getattr(self, 'namespace', None) is not None:
            namespaces['group-specific'] = self.namespace

        # explicit or implicit run namespace
        if run_namespace is not None:
            namespaces['run'] = run_namespace
        else:
            namespaces['implicit'] = get_local_namespace(level+1)

        for description, namespace in namespaces.iteritems():
            if identifier in namespace:
                matches.append((description, namespace[identifier]))

        if len(matches) == 0:
            # No match at all
            raise KeyError(('The identifier "%s" could not be resolved.') %
                           (identifier))
        elif len(matches) > 1:
            # Possibly, all matches refer to the same object
            first_obj = matches[0][1]
            found_mismatch = False
            for m in matches:
                if _same_value(m[1], first_obj):
                    continue
                if _same_function(m[1], first_obj):
                    continue
                try:
                    proxy = weakref.proxy(first_obj)
                    if m[1] is proxy:
                        continue
                except TypeError:
                    pass

                # Found a mismatch
                found_mismatch = True
                break

            if found_mismatch and do_warn:
                _conflict_warning(('The name "%s" refers to different objects '
                                   'in different namespaces used for resolving '
                                   'names in the context of group "%s". '
                                   'Will use the object from the %s namespace '
                                   'with the value %r') %
                                  (identifier, getattr(self, 'name',
                                                       '<unknown>'),
                                   matches[0][0],
                                   first_obj), matches[1:])

        # use the first match (according to resolution order)
        resolved = matches[0][1]

        # Replace pure Python functions by a Functions object
        if callable(resolved) and not isinstance(resolved, Function):
            resolved = Function(resolved)

        if not isinstance(resolved, (Function, Variable)):
            # Wrap the value in a Constant object
            unit = get_unit(resolved)
            value = np.asarray(resolved)
            if value.shape != ():
                raise KeyError('Variable %s was found in the namespace, but is'
                               ' not a scalar value' % identifier)
            resolved = Constant(identifier, unit=unit, value=value)

        return resolved

    def _resolve_all_external(self, identifiers, run_namespace=None, level=0):
        '''
        Parameters
        ----------
        do_warn : int, optional
            Whether to display a warning if an identifier resolves to different
            objects in different namespaces. Defaults to ``True``.
        '''
        resolutions = {}
        for identifier in identifiers:
            resolved = self.resolve(identifier, run_namespace=run_namespace,
                                    level=level+1)
            resolutions[identifier] = resolved

        return resolutions

    def custom_operation(self, code, when=None, name=None):
        '''
        Returns a `CodeRunner` that runs abstract code in the group's namespace.

        Parameters
        ----------
        code : str
            The abstract code to run.
        when : `Scheduler`, optional
            When to run, by default in the 'stateupdate' slot with the same
            clock as the group.
        name : str, optional
            A unique name, if non is given the name of the group appended with
            'custom_operation', 'custom_operation_1', etc. will be used. If a
            name is given explicitly, it will be used as given (i.e. the group
            name will not be prepended automatically).
        '''
        when = Scheduler(when)
        if not when.defined_clock:
            when.clock = self.clock

        if name is None:
            name = self.name + '_custom_operation*'

        runner = CodeRunner(self, 'stateupdate', code=code, name=name,
                            when=when)
        return runner


class CodeRunner(BrianObject):
    '''
    A "code runner" that runs a `CodeObject` every timestep and keeps a
    reference to the `Group`. Used in `NeuronGroup` for `Thresholder`,
    `Resetter` and `StateUpdater`.
    
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
    override_conditional_write: list of str, optional
        A list of variable names which are used as conditions (e.g. for
        refractoriness) which should be ignored.
    '''
    add_to_magic_network = True
    invalidates_magic_network = True
    def __init__(self, group, template, code='', when=None,
                 name='coderunner*', check_units=True, template_kwds=None,
                 needed_variables=None, override_conditional_write=None,
                 ):
        BrianObject.__init__(self, when=when, name=name)
        self.group = weakref.proxy(group)
        self.template = template
        self.abstract_code = code
        self.check_units = check_units
        if needed_variables is None:
            needed_variables = []
        self.needed_variables = needed_variables
        self.template_kwds = template_kwds
        self.override_conditional_write = override_conditional_write
    
    def update_abstract_code(self, run_namespace=None, level=0):
        '''
        Update the abstract code for the code object. Will be called in
        `before_run` and should update the `CodeRunner.abstract_code`
        attribute.
        
        Does nothing by default.
        '''
        pass

    def before_run(self, run_namespace=None, level=0):
        self.update_abstract_code(run_namespace=run_namespace, level=level+1)
        # If the CodeRunner has variables, add them
        if hasattr(self, 'variables'):
            additional_variables = self.variables
        else:
            additional_variables = None

        self.codeobj = create_runner_codeobj(group=self.group,
                                             code=self.abstract_code,
                                             template_name=self.template,
                                             name=self.name+'_codeobject*',
                                             check_units=self.check_units,
                                             additional_variables=additional_variables,
                                             needed_variables=self.needed_variables,
                                             run_namespace=run_namespace,
                                             level=level+1,
                                             template_kwds=self.template_kwds,
                                             override_conditional_write=self.override_conditional_write,
                                             )
        self.code_objects[:] = [weakref.proxy(self.codeobj)]
