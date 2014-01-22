'''
This module defines the `Group` object, a mix-in class for everything that
saves state variables, e.g. `NeuronGroup` or `StateMonitor`.
'''
import weakref

import numpy as np

from brian2.core.base import BrianObject
from brian2.codegen.codeobject import create_runner_codeobj, check_code_units
from brian2.core.variables import AuxiliaryVariable, Variables
from brian2.core.namespace import get_local_namespace
from brian2.units.fundamentalunits import (fail_for_dimension_mismatch, Unit)
from brian2.utils.logger import get_logger
from brian2.devices.device import get_device



__all__ = ['Group', 'GroupCodeRunner']

logger = get_logger(__name__)


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
        self.codeobj = create_runner_codeobj(group=self.group,
                                             code=self.abstract_code,
                                             template_name=self.template,
                                             name=self.name+'_codeobject*',
                                             check_units=self.check_units,
                                             additional_variables=additional_variables,
                                             additional_namespace=namespace,
                                             needed_variables=self.needed_variables,
                                             template_kwds=self.template_kwds)
        self.code_objects[:] = [weakref.proxy(self.codeobj)]

