'''
This module defines the `Group` object, a mix-in class for everything that
saves state variables, e.g. `NeuronGroup` or `StateMonitor`.
'''
import weakref
try:
    from collections import OrderedDict
except ImportError:
    # OrderedDict was added in Python 2.7, use backport for Python 2.6
    from brian2.utils.ordereddict import OrderedDict

import numpy as np

from brian2.core.base import BrianObject
from brian2.codegen.codeobject import create_runner_codeobj, check_code_units
from brian2.core.variables import Variables
from brian2.core.functions import Function
from brian2.core.namespace import (get_local_namespace,
                                   get_default_numpy_namespace,
                                   _get_default_unit_namespace)
from brian2.units.fundamentalunits import (fail_for_dimension_mismatch, Unit)
from brian2.utils.logger import get_logger

__all__ = ['Group', 'GroupCodeRunner']

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
    # use the function itself if it doesn't have a pyfunc attribute and try
    # to create a weak proxy to make a comparison to other weak proxys return
    # true
    func1 = getattr(func1, 'pyfunc', func1)
    try:
        func1 = weakref.proxy(func1)
    except TypeError:
        pass  # already a weakref proxy
    func2 = getattr(func2, 'pyfunc', func2)
    try:
        func2 = weakref.proxy(func2)
    except TypeError:
        pass

    return func1 is func2


class IndexWrapper(object):
    '''
    Convenience class to allow access to the indices via indexing syntax. This
    allows for example to get all indices for synapses originating from neuron
    10 by writing `synapses.indices[10, :]` instead of
    `synapses.calc_indices((10, slice(None))`.
    '''
    def __init__(self, group):
        self.group = weakref.proxy(group)

    def __getitem__(self, item):
        if isinstance(item, basestring):
            namespace = get_local_namespace(1)
            additional_namespace = ('implicit-namespace', namespace)
            variables = Variables(None)
            variables.add_auxiliary_variable('_indices', unit=Unit(1),
                                             dtype=np.int32)
            variables.add_auxiliary_variable('_cond', unit=Unit(1),
                                             dtype=np.bool,
                                             is_bool=True)

            abstract_code = '_cond = ' + item
            check_code_units(abstract_code, self.group,
                             additional_namespace=additional_namespace,
                             additional_variables=variables)
            codeobj = create_runner_codeobj(self.group,
                                            abstract_code,
                                            'group_get_indices',
                                            additional_variables=variables,
                                            additional_namespace=additional_namespace,
                                            )
            return codeobj()
        else:
            return self.group.calc_indices(item)


class Group(BrianObject):
    '''
    Mix-in class for accessing arrays by attribute.
    
    # TODO: Overwrite the __dir__ method to return the state variables
    # (should make autocompletion work)
    '''
    def _enable_group_attributes(self):
        if not hasattr(self, 'variables'):
            raise ValueError('Classes derived from Group need variables attribute.')
        if not hasattr(self, 'codeobj_class'):
            self.codeobj_class = None
        if not hasattr(self, 'indices'):
            self.indices = IndexWrapper(self)

        self._group_attribute_access_active = True

    def state(self, name, use_units, level=0):
        '''
        Return the state variable in a way that properly supports indexing in
        the context of this group

        Parameters
        ----------
        name : str
            The name of the state variable
        use_units : bool
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
            var.get_addressable_value_with_unit(name, self).set_item(slice(None),
                                                                     val,
                                                                     level=1)
        elif len(name) and name[-1]=='_' and name[:-1] in self.variables:
            # no unit checking
            var = self.variables[name[:-1]]
            if var.read_only:
                raise TypeError('Variable %s is read-only.' % name[:-1])
            # Make the call X.var = ... equivalent to X.var[:] = ...
            var.get_addressable_value(name[:-1], self).set_item(slice(None),
                                                                val,
                                                                level=1)
        else:
            object.__setattr__(self, name, val)

    def calc_indices(self, item):
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
        Synapses.calc_indices
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
                    raise TypeError(('Indexing is only supported for integer '
                                     'arrays, not for type %s' % index_array.dtype))
                if index_array.shape == ():
                    index_array = np.array([index_array])
                return index_array

    def resolve(self, identifier, run_namespace=None, level=0):
        '''
        Resolve an external identifier in the context of a `Group`. If the `Group`
        declares an explicit namespace, this namespace is used in addition to the
        standard namespace for units and functions. Additionally, the namespace in
        the `run_namespace` argument (i.e. the namespace provided to `Network.run`)
        is used. Only if both are undefined, the implicit namespace of
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
        strip_unit : bool, optional
            Whether to get rid of units (for the code used in simulation) or not
            (when checking the units of an expression, for example). Defaults to
            ``False``.
        '''
        # We save tuples of (namespace description, referred object) to
        # give meaningful warnings in case of duplicate definitions
        matches = []

        namespaces = OrderedDict()
        namespaces['units'] = _get_default_unit_namespace()
        namespaces['functions'] = get_default_numpy_namespace()
        if getattr(self, 'namespace', None) is not None:
            namespaces['group-specific'] = self.namespace
        if run_namespace is not None:
            namespaces['run'] = run_namespace

        if getattr(self, 'namespace', None) is None and run_namespace is None:
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
                if m[1] is first_obj:
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

            if found_mismatch:
                _conflict_warning(('The name "%s" refers to different objects '
                                   'in different namespaces used for resolving. '
                                   'Will use the object from the %s namespace '
                                   'with the value %r') %
                                  (identifier, matches[0][0],
                                   first_obj), matches[1:])

        # use the first match (according to resolution order)
        resolved = matches[0][1]

        # Replace pure Python functions by a Functions object
        if callable(resolved) and not isinstance(resolved, Function):
            resolved = Function(resolved)

        return resolved

    def resolve_all(self, identifiers, run_namespace=None, level=0):
        resolutions = {}
        for identifier in identifiers:
            resolved = self.resolve(identifier, run_namespace=run_namespace,
                                    level=level+1)
            resolutions[identifier] = resolved

        return resolutions


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
    def __init__(self, group, template, code='', when=None,
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

    def before_run(self, namespace, level=0):
        self.update_abstract_code()
        # If the GroupCodeRunner has variables, add them
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
                                             additional_namespace=namespace,
                                             needed_variables=self.needed_variables,
                                             level=level+1,
                                             template_kwds=self.template_kwds)
        self.code_objects[:] = [weakref.proxy(self.codeobj)]
