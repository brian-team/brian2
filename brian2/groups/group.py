'''
This module defines the `Group` object, a mix-in class for everything that
saves state variables, e.g. `NeuronGroup` or `StateMonitor`.
'''

import weakref

from brian2.core.base import BrianObject
from brian2.core.specifiers import Index
from brian2.units.fundamentalunits import fail_for_dimension_mismatch
from brian2.codegen.translation import analyse_identifiers
from brian2.equations.unitcheck import check_units_statements

__all__ = ['Group', 'GroupCodeRunner']

class Group(object):
    '''
    Mix-in class for accessing arrays by attribute.
    
    # TODO: Overwrite the __dir__ method to return the state variables
    # (should make autocompletion work)
    '''
    def __init__(self):
        if not hasattr(self, 'specifiers'):
            raise ValueError('Classes derived from Group need specifiers attribute')
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
            raise AttributeError

    def __setattr__(self, name, val):
        # attribute access is switched off until this attribute is created by
        # Group.__init__
        if not hasattr(self, '_group_attribute_access_active'):
            object.__setattr__(self, name, val)
        elif name in self.specifiers:
            spec = self.specifiers[name]
            fail_for_dimension_mismatch(val, spec.unit,
                                        'Incorrect units for setting %s' % name)
            spec.set_value(val)
        elif len(name) and name[-1]=='_' and name[:-1] in self.specifiers:
            # no unit checking
            self.specifiers[name[:-1]].set_value(val)
        else:
            object.__setattr__(self, name, val)


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
                 when=None, name='coderunner*', check_units=True,
                 template_specifiers=None):
        BrianObject.__init__(self, when=when, name=name)
        self.group = weakref.proxy(group)
        self.template = template
        self.indices = indices
        self.abstract_code = code
        self.check_units = check_units
        self.template_specifiers = template_specifiers
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

        all_specifiers = dict(self.group.specifiers)
        # If the GroupCodeRunner has specifiers, add them
        if hasattr(self, 'specifiers'):
            all_specifiers.update(self.specifiers)
                        
        if self.check_units:
            # Resolve the namespace, resulting in a dictionary containing only the
            # external variables that are needed by the code -- keep the units for
            # the unit checks 
            _, _, unknown = analyse_identifiers(self.abstract_code,
                                                all_specifiers.keys())
            resolved_namespace = self.group.namespace.resolve_all(unknown,
                                                                  additional_namespace,
                                                                  strip_units=False)
        
            check_units_statements(self.abstract_code, resolved_namespace,
                                   all_specifiers)
    
        # Get the namespace without units
        _, used_known, unknown = analyse_identifiers(self.abstract_code,
                                                     all_specifiers.keys())
        resolved_namespace = self.group.namespace.resolve_all(unknown,
                                                              additional_namespace)
        
        # Only pass the specifiers that are actually used
        specifiers = {}
        for name in used_known:
            specifiers[name] = all_specifiers[name]
        
        if self.template_specifiers:
            for spec in self.template_specifiers:
                specifiers[spec] = all_specifiers[spec]
        
        return self.group.language.create_codeobj(self.name,
                                             self.abstract_code,
                                             resolved_namespace,
                                             specifiers,
                                             self.template,
                                             indices=self.indices)
    
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
