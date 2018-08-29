'''
This module defines the `VariableOwner` class, a mix-in class for everything
that saves state variables, e.g. `Clock` or `NeuronGroup`, the class `Group`
for objects that in addition to storing state variables also execute code, i.e.
objects such as `NeuronGroup` or `StateMonitor` but not `Clock`, and finally
`CodeRunner`, a class to run code in the context of a `Group`.
'''
import collections
from collections import OrderedDict
import weakref
import numbers
import inspect

import numpy as np

from brian2.core.base import BrianObject, weakproxy_with_fallback
from brian2.core.names import Nameable
from brian2.core.preferences import prefs
from brian2.core.variables import (Variables, Constant, Variable,
                                   ArrayVariable, DynamicArrayVariable,
                                   Subexpression, AuxiliaryVariable)
from brian2.core.functions import Function
from brian2.core.namespace import (get_local_namespace,
                                   DEFAULT_FUNCTIONS,
                                   DEFAULT_UNITS,
                                   DEFAULT_CONSTANTS)
from brian2.codegen.codeobject import create_runner_codeobj
from brian2.codegen.generators.numpy_generator import NumpyCodeGenerator
from brian2.equations.equations import BOOLEAN, INTEGER, FLOAT, Equations
from brian2.units.fundamentalunits import (fail_for_dimension_mismatch, Unit,
                                           get_unit, DIMENSIONLESS)
from brian2.utils.logger import get_logger
from brian2.utils.stringtools import get_identifiers, SpellChecker
from brian2.importexport.importexport import ImportExport

__all__ = ['Group', 'VariableOwner', 'CodeRunner']

logger = get_logger(__name__)


def _display_value(obj):
    '''
    Helper function for warning messages that display the value of objects. This
    functions returns a nicer representation for symbolic constants and
    functions.

    Parameters
    ----------
    obj : object
        The object to display

    Returns
    -------
    value : str
        A string representation of the object
    '''
    if isinstance(obj, Function):
        return '<Function>'
    try:
        obj = obj.get_value()
    except AttributeError:
        pass
    try:
        obj = obj.value
    except AttributeError:
        pass

    # We (temporarily) set numpy's print options so that array with more than
    # 10 elements are only shown in an abbreviated way
    old_options = np.get_printoptions()
    np.set_printoptions(threshold=10)
    try:
        str_repr = repr(obj)
    except Exception:
        str_repr = '<object of type %s>' % type(obj)
    finally:
        np.set_printoptions(**old_options)
    return str_repr

def _conflict_warning(message, resolutions):
    '''
    A little helper functions to generate warnings for logging. Specific
    to the `Group._resolve` method and should only be used by it.

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
        second_part = ('but the name also refers to a variable in the %s '
                       'namespace with value %s.') % (resolutions[0][0],
                                                      _display_value(resolutions[0][1]))
    else:
        second_part = ('but the name also refers to a variable in the following '
                       'namespaces: %s.') % (', '.join([r[0]
                                                        for r in resolutions]))

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
        return prefs['core.default_integer_dtype']
    elif equation.var_type == FLOAT:
        if dtype is not None:
            dtype = np.dtype(dtype)
            if not dtype.kind == 'f':
                raise TypeError(('%s is not a valid floating point '
                                 'dtype') % dtype)
            return dtype
        else:
            return prefs['core.default_float_dtype']
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
    when the respective `VariableOwner` no longer exists. Note that this object
    does not handle string indexing.
    '''
    def __init__(self, group, default_index='_idx'):
        self.group = weakref.proxy(group)
        self.N = group.variables['N']
        self.default_index = default_index

    def __call__(self, item=slice(None), index_var=None):
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
        if index_var is None:
            index_var = self.default_index

        if hasattr(item, '_indices'):
            item = item._indices()

        if isinstance(item, tuple):
            raise IndexError(('Can only interpret 1-d indices, '
                              'got %d dimensions.') % len(item))
        else:
            if isinstance(item, basestring) and item == 'True':
                item = slice(None)
            if isinstance(item, slice):
                if index_var == '0':
                    return 0
                if index_var == '_idx':
                    start, stop, step = item.indices(int(self.N.get_value()))
                else:
                    start, stop, step = item.indices(index_var.size)
                index_array = np.arange(start, stop, step, dtype=np.int32)
            else:
                index_array = np.asarray(item)
                if index_array.dtype == np.bool:
                    index_array = np.nonzero(index_array)[0]
                elif not np.issubdtype(index_array.dtype, np.signedinteger):
                    raise TypeError(('Indexing is only supported for integer '
                                     'and boolean arrays, not for type '
                                     '%s' % index_array.dtype))

            if index_var != '_idx':
                try:
                    return index_var.get_value()[index_array]
                except IndexError as ex:
                    # We try to emulate numpy's indexing semantics here:
                    # slices never lead to IndexErrors, instead they return an
                    # empty array if they don't match anything
                    if isinstance(item, slice):
                        return np.array([], dtype=np.int32)
                    else:
                        raise ex
            else:
                return index_array


class IndexWrapper(object):
    '''
    Convenience class to allow access to the indices via indexing syntax. This
    allows for example to get all indices for synapses originating from neuron
    10 by writing `synapses.indices[10, :]` instead of
    `synapses._indices.((10, slice(None))`.
    '''
    def __init__(self, group):
        self.group = weakref.proxy(group)
        self.indices = group._indices

    def __getitem__(self, item):
        if isinstance(item, basestring):
            variables = Variables(None)
            variables.add_auxiliary_variable('_indices', dtype=np.int32)
            variables.add_auxiliary_variable('_cond', dtype=np.bool)

            abstract_code = '_cond = ' + item
            namespace = get_local_namespace(level=1)
            from brian2.devices.device import get_device
            device = get_device()
            codeobj = create_runner_codeobj(self.group,
                                            abstract_code,
                                            'group_get_indices',
                                            run_namespace=namespace,
                                            additional_variables=variables,
                                            codeobj_class=device.code_object_class(fallback_pref='codegen.string_expression_target')
                                            )
            return codeobj()
        else:
            return self.indices(item)


class VariableOwner(Nameable):
    '''
    Mix-in class for accessing arrays by attribute.

    # TODO: Overwrite the __dir__ method to return the state variables
    # (should make autocompletion work)
    '''
    def _enable_group_attributes(self):
        if not hasattr(self, 'variables'):
            raise ValueError(('Classes derived from VariableOwner need a '
                              'variables attribute.'))
        if not 'N' in self.variables:
            raise ValueError('Each VariableOwner needs an "N" variable.')
        if not hasattr(self, 'codeobj_class'):
            self.codeobj_class = None
        if not hasattr(self, '_indices'):
            self._indices = Indexing(self)
        if not hasattr(self, 'indices'):
            self.indices = IndexWrapper(self)
        if not hasattr(self, '_stored_states'):
            self._stored_states = {}
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
        if not '_group_attribute_access_active' in self.__dict__:
            raise AttributeError
        if (name in self.__getattribute__('__dict__') or
                    name in self.__getattribute__('__class__').__dict__):
            # Makes sure that classes can override the "variables" mechanism
            # with instance/class attributes and properties
            return object.__getattribute__(self, name)
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
        elif (name in self.__getattribute__('__dict__') or
                    name in self.__getattribute__('__class__').__dict__):
            # Makes sure that classes can override the "variables" mechanism
            # with instance/class attributes and properties
            return object.__setattr__(self, name, val)
        elif name in self.variables:
            var = self.variables[name]
            if not isinstance(val, basestring):
                if var.dim is DIMENSIONLESS:
                    fail_for_dimension_mismatch(val, var.dim,
                                                ('%s should be set with a '
                                                 'dimensionless value, but got '
                                                 '{value}') % name,
                                                value=val)
                else:
                    fail_for_dimension_mismatch(val, var.dim,
                                                ('%s should be set with a '
                                                 'value with units %r, but got '
                                                 '{value}') % (name, get_unit(var.dim)),
                                                value=val)
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
        elif hasattr(self, name) or name.startswith('_'):
            object.__setattr__(self, name, val)
        else:
            # Try to suggest the correct name in case of a typo
            checker = SpellChecker([varname for varname, var in self.variables.iteritems()
                                    if not (varname.startswith('_') or var.read_only)])
            if name.endswith('_'):
                suffix = '_'
                name = name[:-1]
            else:
                suffix = ''
            error_msg = 'Could not find a state variable with name "%s".' % name
            suggestions = checker.suggest(name)
            if len(suggestions) == 1:
                suggestion, = suggestions
                error_msg += ' Did you mean to write "%s%s"?' % (suggestion,
                                                                 suffix)
            elif len(suggestions) > 1:
                error_msg += (' Did you mean to write any of the following: %s ?' %
                              (', '.join(['"%s%s"' % (suggestion, suffix)
                                          for suggestion in suggestions])))
            error_msg += (' Use the add_attribute method if you intend to add '
                          'a new attribute to the object.')
            raise AttributeError(error_msg)

    def add_attribute(self, name):
        '''
        Add a new attribute to this group. Using this method instead of simply
        assigning to the new attribute name is necessary because Brian will
        raise an error in that case, to avoid bugs passing unnoticed
        (misspelled state variable name, un-declared state variable, ...).

        Parameters
        ----------
        name : str
            The name of the new attribute

        Raises
        ------
        AttributeError
            If the name already exists as an attribute or a state variable.
        '''
        if name in self.variables:
            raise AttributeError('Cannot add an attribute "%s", it is already '
                                 'a state variable of this group.' % name)
        if hasattr(self, name):
            raise AttributeError('Cannot add an attribute "%s", it is already '
                                 'an attribute of this group.' % name)
        object.__setattr__(self, name, None)

    def get_states(self, vars=None, units=True, format='dict',
                   subexpressions=False, read_only_variables=True, level=0):
        '''
        Return a copy of the current state variable values. The returned arrays
        are copies of the actual arrays that store the state variable values,
        therefore changing the values in the returned dictionary will not affect
        the state variables.

        Parameters
        ----------
        vars : list of str, optional
            The names of the variables to extract. If not specified, extract
            all state variables (except for internal variables, i.e. names that
            start with ``'_'``). If the ``subexpressions`` argument is ``True``,
            the current values of all subexpressions are returned as well.
        units : bool, optional
            Whether to include the physical units in the return value. Defaults
            to ``True``.
        format : str, optional
            The output format. Defaults to ``'dict'``.
        subexpressions: bool, optional
            Whether to return subexpressions when no list of variable names
            is given. Defaults to ``False``. This argument is ignored if an
            explicit list of variable names is given in ``vars``.
        read_only_variables : bool, optional
            Whether to return read-only variables (e.g. the number of neurons,
            the time, etc.). Setting it to ``False`` will assure that the
            returned state can later be used with `set_states`. Defaults to
            ``True``.
        level : int, optional
            How much higher to go up the stack to resolve external variables.
            Only relevant if extracting subexpressions that refer to external
            variables.

        Returns
        -------
        values : dict or specified format
            The variables specified in ``vars``, in the specified ``format``.

        '''
        if format not in ImportExport.methods:
            raise NotImplementedError("Format '%s' is not supported" % format)
        if vars is None:
            vars = []
            for name, var in self.variables.iteritems():
                if name.startswith('_'):
                    continue
                if subexpressions or not isinstance(var, Subexpression):
                    if read_only_variables or not getattr(var, 'read_only', False):
                        if not isinstance(var, AuxiliaryVariable):
                            vars.append(name)
        data = ImportExport.methods[format].export_data(self, vars, units=units, level=level)
        return data

    def set_states(self, values, units=True, format='dict', level=0):
        '''
        Set the state variables.

        Parameters
        ----------
        values : depends on ``format``
            The values according to ``format``.
        units : bool, optional
            Whether the ``values`` include physical units. Defaults to ``True``.
        format : str, optional
            The format of ``values``. Defaults to ``'dict'``
        level : int, optional
            How much higher to go up the stack to resolve external variables.
            Only relevant when using string expressions to set values.
        '''
        # For the moment, 'dict' is the only supported format -- later this will
        # be made into an extensible system, see github issue #306
        if format not in ImportExport.methods:
            raise NotImplementedError("Format '%s' is not supported" % format)
        ImportExport.methods[format].import_data(self, values, units=units, level=level)

    def check_variable_write(self, variable):
        '''
        Function that can be overwritten to raise an error if writing to a
        variable should not be allowed. Note that this does *not* deal with
        incorrect writes that are general to all kind of variables (incorrect
        units, writing to a read-only variable, etc.). This function is only
        used for type-specific rules, e.g. for raising an error in `Synapses`
        when writing to a synaptic variable before any `~Synapses.connect`
        call.

        By default this function does nothing.

        Parameters
        ----------
        variable : `Variable`
            The variable that the user attempts to set.
        '''
        pass

    def _full_state(self):
        state = {}
        for var in self.variables.itervalues():
            if not isinstance(var, ArrayVariable):
                continue  # we are only interested in arrays
            if var.owner is None or var.owner.name != self.name:
                continue  # we only store the state of our own variables

            state[var.name] = (var.get_value().copy(), var.size)

        return state

    def _restore_from_full_state(self, state):
        for var_name, (values, size) in state.iteritems():
            var = self.variables[var_name]
            if isinstance(var, DynamicArrayVariable):
                var.resize(size)
            var.set_value(values)

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

    def __len__(self):
        return int(self.variables['N'].get_value())


class Group(VariableOwner, BrianObject):

    def _resolve(self, identifier, run_namespace, user_identifier=True,
                 additional_variables=None):
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
        run_namespace : dict-like, optional
            An additional namespace that is used for variable lookup (if not
            defined, the implicit namespace of local variables is used).
        user_identifier : bool, optional
            Whether this is an identifier that was used by the user (and not
            something automatically generated that the user might not even
            know about). Will be used to determine whether to display a
            warning in the case of namespace clashes. Defaults to ``True``.
        additional_variables : dict-like, optional
            An additional mapping of names to `Variable` objects that will be
            checked before `Group.variables`.

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
            if not user_identifier:
                return resolved_internal  # no need to go further
            # We already found the identifier, but we try to resolve it in the
            # external namespace nevertheless, to report a warning if it is
            # present there as well.
            self._resolve_external(identifier, run_namespace=run_namespace,
                                   internal_variable=resolved_internal)
            return resolved_internal

        # We did not find the name internally, try to resolve it in the external
        # namespace
        return self._resolve_external(identifier, run_namespace=run_namespace)

    def resolve_all(self, identifiers, run_namespace, user_identifiers=None,
                    additional_variables=None):
        '''
        Resolve a list of identifiers. Calls `Group._resolve` for each
        identifier.

        Parameters
        ----------
        identifiers : iterable of str
            The names to look up.
        run_namespace : dict-like, optional
            An additional namespace that is used for variable lookup (if not
            defined, the implicit namespace of local variables is used).
        user_identifiers : iterable of str, optional
            The names in ``identifiers`` that were provided by the user (i.e.
            are part of user-specified equations, abstract code, etc.). Will
            be used to determine when to issue namespace conflict warnings. If
            not specified, will be assumed to be identical to ``identifiers``.
        additional_variables : dict-like, optional
            An additional mapping of names to `Variable` objects that will be
            checked before `Group.variables`.

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
        if user_identifiers is None:
            user_identifiers = identifiers
        assert isinstance(run_namespace, collections.Mapping)
        resolved = {}
        for identifier in identifiers:
            resolved[identifier] = self._resolve(identifier,
                                                 user_identifier=identifier in user_identifiers,
                                                 additional_variables=additional_variables,
                                                 run_namespace=run_namespace)
        return resolved

    def _resolve_external(self, identifier, run_namespace, user_identifier=True,
                          internal_variable=None):
        '''
        Resolve an external identifier in the context of a `Group`. If the `Group`
        declares an explicit namespace, this namespace is used in addition to the
        standard namespace for units and functions. Additionally, the namespace in
        the `run_namespace` argument (i.e. the namespace provided to `Network.run`)
        is used.

        Parameters
        ----------
        identifier : str
            The name to resolve.
        group : `Group`
            The group that potentially defines an explicit namespace for looking up
            external names.
        run_namespace : dict
            A namespace (mapping from strings to objects), as provided as an
            argument to the `Network.run` function or returned by
            `get_local_namespace`.
        user_identifier : bool, optional
            Whether this is an identifier that was used by the user (and not
            something automatically generated that the user might not even
            know about). Will be used to determine whether to display a
            warning in the case of namespace clashes. Defaults to ``True``.
        internal_variable : `Variable`, optional
            The internal variable object that corresponds to this name (if any).
            This is used to give warnings if it also corresponds to a variable
            from an external namespace.
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
        namespaces['run'] = run_namespace

        for description, namespace in namespaces.iteritems():
            if identifier in namespace:
                match = namespace[identifier]
                if ((isinstance(match, (numbers.Number,
                                         np.ndarray,
                                         np.number,
                                         Function,
                                         Variable))) or
                         (inspect.isfunction(match) and
                              hasattr(match, '_arg_units') and
                              hasattr(match, '_return_unit'))
                    ):
                    matches.append((description, match))

        if len(matches) == 0:
            # No match at all
            if internal_variable is not None:
                return None
            else:
                # Give a more detailed explanation for the lastupdate variable
                # that was removed with PR #1003
                if identifier == 'lastupdate':
                    error_msg = ('The identifier "lastupdate" could not be '
                                 'resolved. Note that this variable is only '
                                 'automatically defined for models with '
                                 'event-driven synapses. You can define it '
                                 'manually by adding "lastupdate : second" to '
                                 'the equations and setting "lastupdate = t" '
                                 'at the end of your on_pre and/or on_post '
                                 'statements.')
                else:
                    error_msg = ('The identifier "%s" could not be resolved.' %
                                 identifier)
                raise KeyError(error_msg)

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

            if found_mismatch and user_identifier and internal_variable is None:
                _conflict_warning(('The name "%s" refers to different objects '
                                   'in different namespaces used for resolving '
                                   'names in the context of group "%s". '
                                   'Will use the object from the %s namespace '
                                   'with the value %s,') %
                                  (identifier, getattr(self, 'name',
                                                       '<unknown>'),
                                   matches[0][0],
                                   _display_value(first_obj)), matches[1:])

        if internal_variable is not None and user_identifier:
            # Filter out matches that are identical (a typical case being an
            # externally defined "N" with the the number of neurons and a later
            # use of "N" in an expression (which refers to the internal variable
            # storing the number of neurons in the group)
            if isinstance(internal_variable, Constant):
                filtered_matches = []
                for match in matches:
                    if not _same_value(match[1], internal_variable):
                        filtered_matches.append(match)
            else:
                filtered_matches = matches
            if len(filtered_matches) == 0:
                pass  # Nothing to warn about
            else:
                warning_message = ('"{name}" is an internal variable of group '
                                   '"{group}", but also exists in the ')
                if len(matches) == 1:
                    warning_message += ('{namespace} namespace with the value '
                                        '{value}. ').format(namespace=filtered_matches[0][0],
                                                           value=_display_value(filtered_matches[0][1]))
                else:
                    warning_message += ('following namespaces: '
                                        '{namespaces}. ').format(namespaces=' ,'.join(match[0]
                                                                                     for match in filtered_matches))
                warning_message += 'The internal variable will be used.'
                logger.warn(warning_message.format(name=identifier,
                                                   group=self.name),
                            'Group.resolve.resolution_conflict', once=True)

        if internal_variable is not None:
            return None  # We were only interested in the warnings above

        # use the first match (according to resolution order)
        resolved = matches[0][1]

        # Replace pure Python functions by a Functions object
        if callable(resolved) and not isinstance(resolved, Function):
            resolved = Function(resolved,
                                arg_units=getattr(resolved, '_arg_units', None),
                                return_unit=getattr(resolved, '_return_unit', None),
                                stateless=getattr(resolved, 'stateless', False))

        if not isinstance(resolved, (Function, Variable)):
            # Wrap the value in a Constant object
            dimensions = getattr(resolved, 'dim', DIMENSIONLESS)
            value = np.asarray(resolved)
            if value.shape != ():
                raise KeyError('Variable %s was found in the namespace, but is'
                               ' not a scalar value' % identifier)
            resolved = Constant(identifier, dimensions=dimensions, value=value)

        return resolved

    def runner(self, *args, **kwds):
        raise AttributeError("The 'runner' method has been renamed to "
                             "'run_regularly'.")

    def custom_operation(self, *args, **kwds):
        raise AttributeError("The 'custom_operation' method has been renamed "
                             "to 'run_regularly'.")

    def run_regularly(self, code, dt=None, clock=None, when='start',
                      order=0, name=None, codeobj_class=None):
        '''
        Run abstract code in the group's namespace. The created `CodeRunner`
        object will be automatically added to the group, it therefore does not
        need to be added to the network manually. However, a reference to the
        object will be returned, which can be used to later remove it from the
        group or to set it to inactive.

        Parameters
        ----------
        code : str
            The abstract code to run.
        dt : `Quantity`, optional
            The time step to use for this custom operation. Cannot be combined
            with the `clock` argument.
        clock : `Clock`, optional
            The update clock to use for this operation. If neither a clock nor
            the `dt` argument is specified, defaults to the clock of the group.
        when : str, optional
            When to run within a time step, defaults to the ``'start'`` slot.
        name : str, optional
            A unique name, if non is given the name of the group appended with
            'run_regularly', 'run_regularly_1', etc. will be used. If a
            name is given explicitly, it will be used as given (i.e. the group
            name will not be prepended automatically).
        codeobj_class : class, optional
            The `CodeObject` class to run code with. If not specified, defaults
            to the `group`'s ``codeobj_class`` attribute.

        Returns
        -------
        obj : `CodeRunner`
            A reference to the object that will be run.
        '''
        if name is None:
            name = self.name + '_run_regularly*'

        if dt is None and clock is None:
            clock = self._clock

        # Subgroups are normally not included in their parent's
        # contained_objects list, since there's no need to include them in the
        # network (they don't perform any computation on their own). However,
        # if a subgroup declares a `run_regularly` operation, then we want to
        # include this operation automatically, i.e. with the parent group
        # (adding just the run_regularly operation to the parent group's
        # contained objects would no be enough, since the CodeObject needs a
        # reference to the group providing the context for the operation, i.e.
        # the subgroup instead of the parent group. See github issue #922
        source_group = getattr(self, 'source', None)
        if source_group is not None:
            if not self in source_group.contained_objects:
                source_group.contained_objects.append(self)

        runner = CodeRunner(self, 'stateupdate', code=code, name=name,
                            dt=dt, clock=clock, when=when, order=order,
                            codeobj_class=codeobj_class)
        self.contained_objects.append(runner)
        return runner

    def _check_for_invalid_states(self):
        '''
        Checks if any state variables updated by differential equations have
        invalid values, and logs a warning if so.
        '''
        equations = getattr(self, 'equations', None)
        if not isinstance(equations, Equations):
            return
        for varname in equations.diff_eq_names:
            self._check_for_invalid_values(varname, self.state(varname,
                                                               use_units=False))

    def _check_for_invalid_values(self, k, v):
        '''
        Checks if variable named k value v has invalid values, and logs a
        warning if so.
        '''
        v = np.asarray(v)
        if np.isnan(v).any() or (np.abs(v) > 1e50).any():
            logger.warn(("{name}'s variable '{k}' has NaN, very large values, "
                         "or encountered an error in numerical integration. "
                         "This is usually a sign that an unstable or invalid "
                         "integration method was "
                         "chosen.").format(name=self.name,
                                           k=k),
                        name_suffix="invalid_values", once=True)


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
    dt : `Quantity`, optional
        The time step to be used for the simulation. Cannot be combined with
        the `clock` argument.
    user_code : str, optional
        The abstract code as specified by the user, i.e. without any additions
        of internal code that the user not necessarily knows about. This will
        be used for warnings and error messages.
    clock : `Clock`, optional
        The update clock to be used. If neither a clock, nor the `dt` argument
        is specified, the `defaultclock` will be used.
    when : str, optional
        In which scheduling slot to execute the operation during a time step.
        Defaults to ``'start'``.
    order : int, optional
        The priority of this operation for operations occurring at the same time
        step and in the same scheduling slot. Defaults to 0.
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
    codeobj_class : class, optional
        The `CodeObject` class to run code with. If not specified, defaults to
        the `group`'s ``codeobj_class`` attribute.
    generate_empty_code : bool, optional
        Whether to generate a `CodeObject` if there is no abstract code to
        execute. Defaults to ``True`` but should be switched off e.g. for a
        `StateUpdater` when there is nothing to do.
    '''
    add_to_magic_network = True
    invalidates_magic_network = True
    def __init__(self, group, template, code='', user_code=None,
                 dt=None, clock=None, when='start',
                 order=0, name='coderunner*', check_units=True,
                 template_kwds=None, needed_variables=None,
                 override_conditional_write=None,
                 codeobj_class=None,
                 generate_empty_code=True
                 ):
        BrianObject.__init__(self, clock=clock, dt=dt, when=when, order=order,
                             name=name)
        self.group = weakproxy_with_fallback(group)
        self.template = template
        self.user_code = user_code
        self.abstract_code = code
        self.check_units = check_units
        if needed_variables is None:
            needed_variables = []
        self.needed_variables = needed_variables
        self.template_kwds = template_kwds
        self.override_conditional_write = override_conditional_write
        if codeobj_class is None:
            codeobj_class = group.codeobj_class
        self.codeobj_class = codeobj_class
        self.generate_empty_code = generate_empty_code
        self.codeobj = None

    def update_abstract_code(self, run_namespace):
        '''
        Update the abstract code for the code object. Will be called in
        `before_run` and should update the `CodeRunner.abstract_code`
        attribute.
        
        Does nothing by default.
        '''
        pass

    def before_run(self, run_namespace):
        self.update_abstract_code(run_namespace=run_namespace)
        # If the CodeRunner has variables, add them
        if hasattr(self, 'variables'):
            additional_variables = self.variables
        else:
            additional_variables = None

        if not self.generate_empty_code and len(self.abstract_code) == 0:
            self.codeobj = None
            self.code_objects[:] = []
        else:
            self.codeobj = create_runner_codeobj(group=self.group,
                                                 code=self.abstract_code,
                                                 user_code=self.user_code,
                                                 template_name=self.template,
                                                 name=self.name+'_codeobject*',
                                                 check_units=self.check_units,
                                                 additional_variables=additional_variables,
                                                 needed_variables=self.needed_variables,
                                                 run_namespace=run_namespace,
                                                 template_kwds=self.template_kwds,
                                                 override_conditional_write=self.override_conditional_write,
                                                 codeobj_class=self.codeobj_class
                                                 )
            self.code_objects[:] = [weakref.proxy(self.codeobj)]
