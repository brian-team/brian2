'''
This module defines the `Group` object, a mix-in class for everything that
saves state variables, e.g. `NeuronGroup` or `StateMonitor`.
'''
import weakref

import numpy as np

from brian2.core.base import BrianObject
from brian2.core.specifiers import ArrayVariable, Index
from brian2.core.namespace import get_local_namespace
from brian2.units.fundamentalunits import fail_for_dimension_mismatch, Unit
from brian2.codegen.translation import analyse_identifiers
from brian2.equations.unitcheck import check_units_statements

__all__ = ['Group', 'GroupCodeRunner', 'Indices']


class Indices(object):

    def __init__(self, N):
        self.N = N
        self._indices = np.arange(self.N)
        self.specifiers = {'i': ArrayVariable('i',
                                              Unit(1),
                                              self._indices.dtype,
                                              self._indices,
                                              '_neuron_idx')}

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
            namespace = {'i': self._indices}

            result = eval(index, namespace)
            return np.flatnonzero(result)
        else:
            return self._indices[index]


class Group(object):
    '''
    Mix-in class for accessing arrays by attribute.
    
    # TODO: Overwrite the __dir__ method to return the state variables
    # (should make autocompletion work)
    '''
    def __init__(self):
        if not hasattr(self, 'specifiers'):
            raise ValueError('Classes derived from Group need specifiers attribute.')
        if not hasattr(self, 'indices'):
            try:
                N = len(self)
            except TypeError:
                raise ValueError(('Classes derived from Group need an indices '
                                  'attribute, or a length to automatically '
                                  'provide 1-d indexing'))
            self.indices = Indices(N)
        self._group_attribute_access_active = True
    
    def state_(self, name):
        '''
        Gets the unitless array.
        '''
        try:
            return self.specifiers[name].get_addressable_value()
        except KeyError:
            raise KeyError("Array named "+name+" not found.")
        
    def state(self, name):
        '''
        Gets the array with units.
        '''
        try:
            spec = self.specifiers[name]
            return spec.get_addressable_value_with_unit()
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
        elif name in self.specifiers:
            spec = self.specifiers[name]
            if not isinstance(val, basestring):
                fail_for_dimension_mismatch(val, spec.unit,
                                            'Incorrect units for setting %s' % name)
            # Make the call X.var = ... equivalent to X.var[:] = ...
            spec.get_addressable_value_with_unit(level=1)[:] = val
        elif len(name) and name[-1]=='_' and name[:-1] in self.specifiers:
            # no unit checking
            spec = self.specifiers[name[:-1]]
            # Make the call X.var = ... equivalent to X.var[:] = ...
            spec.get_addressable_value(level=1)[:] = val
        else:
            object.__setattr__(self, name, val)

    def update_namespace(self):
        pass

    def _set_with_code(self, specifier, group_indices, code,
                       check_units=True, level=0):
        '''
        Sets a variable using a string expression. Is called by
        `VariableView.__setitem__` for statements such as
        `S.var[:, :] = 'exp(-abs(i-j)/space_constant)*nS'`

        Parameters
        ----------
        specifier : `ArrayVariable`
            The `Specifier` for the variable to be set
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
        self.update_namespace()

        abstract_code = specifier.name + ' = ' + code
        indices = {'_neuron_idx': Index('_neuron_idx', iterate_all=False)}
        namespace = get_local_namespace(level + 1)
        additional_namespace = ('implicit-namespace', namespace)
        additional_specifiers = dict(self.indices.specifiers)
        # TODO: Find a name that makes sense for reset and variable setting
        # with code
        additional_specifiers['_spikes'] = ArrayVariable('_spikes',
                                                         Unit(1),
                                                         np.int32,
                                                         group_indices,
                                                         '',  # no index,
                                                         self)
        codeobj = create_codeobj(self,
                                 abstract_code,
                                 self.language.template_reset,
                                 indices,
                                 additional_specifiers=additional_specifiers,
                                 additional_namespace=additional_namespace,
                                 check_units=check_units,
                                 language=self.language)
        codeobj()

def create_codeobj(group, code, template, indices,
                   name=None, check_units=True, additional_specifiers=None,
                   additional_namespace=None, language=None):
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
    indices : dict-like
        A mapping from index name to `Index` objects, describing the indices
        used for the variables in the code.
    name : str, optional
        A name for this code object, will use ``group + '_codeobject*'`` if
        none is given.
    check_units : bool, optional
        Whether to check units in the statement. Defaults to ``True``.
    additional_specifiers : dict-like, optional
        A mapping of names to `Specifier` objects, used in addition to the
        specifiers saved in `group`.
    additional_namespace : dict-like, optional
        A mapping from names to objects, used in addition to the namespace
        saved in `group`.
    '''
    if group is not None:
        all_specifiers = dict(group.specifiers)
    else:
        all_specifiers = {}
    # If the GroupCodeRunner has specifiers, add them
    if additional_specifiers is not None:
        all_specifiers.update(additional_specifiers)

    if check_units:
        # Resolve the namespace, resulting in a dictionary containing only the
        # external variables that are needed by the code -- keep the units for
        # the unit checks
        _, _, unknown = analyse_identifiers(code,
                                            all_specifiers.keys())
        if group is not None:
            resolved_namespace = group.namespace.resolve_all(unknown,
                                                             additional_namespace,
                                                             strip_units=False)
        else:
            resolved_namespace = {}

        check_units_statements(code, resolved_namespace, all_specifiers)

    # Get the namespace without units
    _, used_known, unknown = analyse_identifiers(code, all_specifiers.keys())
    if group is not None:
        resolved_namespace = group.namespace.resolve_all(unknown,
                                                         additional_namespace)
    else:
        resolved_namespace = {}

    # Only pass the specifiers that are actually used
    specifiers = {}
    for name in used_known:
        specifiers[name] = all_specifiers[name]

    # Also add the specifiers that the template needs
    for spec in template.specifiers:
        try:
            specifiers[spec] = all_specifiers[spec]
        except KeyError as ex:
            # We abuse template.specifiers here to also store names of things
            # from the namespace (e.g. rand) that are needed
            # TODO: Improve all of this namespace/specifier handling
            if group is not None:
                # Try to find the name in the group's namespace
                resolved_namespace[spec] = group.namespace.resolve(spec,
                                                                   additional_namespace)
            else:
                raise ex

    if name is None:
        if group is not None:
            name = group.name + '_codeobject*'
        else:
            name = '_codeobject*'

    return language.create_codeobj(name,
                                   code,
                                   resolved_namespace,
                                   specifiers,
                                   template,
                                   indices=indices)


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
    iterate_all : bool, optional
        Whether the index iterates over all possible values (``True``, the
        default) or only over a subset (``False``, used for example for the
        reset which only affects neurons that have spiked).
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
    template_specifiers : list of str, optional
        A list of specifiers that are needed in the template even though they
        are not present in the abstract code. TODO: This information should
        be represented in the template itself.
    
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
    def __init__(self, group, template, indices, code=None,
                 when=None, name='coderunner*', check_units=True):
        BrianObject.__init__(self, when=when, name=name)
        self.group = weakref.proxy(group)
        self.template = template
        self.indices = indices
        self.abstract_code = code
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
        '''
        Update the abstract code for the code object. Will be called in
        `pre_run` and should update the `GroupCodeRunner.abstract_code`
        attribute.
        
        Does nothing by default.
        '''
        pass

    def _create_codeobj(self, additional_namespace=None):
        ''' A little helper function to reduce the amount of repetition when
        calling the language's _create_codeobj (always pass self.specifiers and
        self.namespace + additional namespace).
        '''

        # If the GroupCodeRunner has specifiers, add them
        if hasattr(self, 'specifiers'):
            additional_specifiers = self.specifiers
        else:
            additional_specifiers = None

        return create_codeobj(self.group, self.abstract_code, self.template,
                              self.indices, self.name, self.check_units,
                              additional_specifiers=additional_specifiers,
                              additional_namespace=additional_namespace,
                              language=self.group.language)
    
    def pre_run(self, namespace):
        self.update_abstract_code()
        self.codeobj = self._create_codeobj(additional_namespace=namespace)
    
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
