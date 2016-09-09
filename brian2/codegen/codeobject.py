'''
Module providing the base `CodeObject` and related functions.
'''
import copy
import platform
import weakref

from brian2.core.names import Nameable
from brian2.equations.unitcheck import check_units_statements
from brian2.utils.logger import get_logger
from brian2.utils.stringtools import indent, code_representation

from .translation import analyse_identifiers

__all__ = ['CodeObject',
           'CodeObjectUpdater',
           'constant_or_scalar']

logger = get_logger(__name__)


#: Dictionary with basic information about the current system (OS, etc.)
sys_info = {'system': platform.system(),
            'architecture': platform.architecture(),
            'machine': platform.machine()}


def constant_or_scalar(varname, variable):
    '''
    Convenience function to generate code to access the value of a variable.
    Will return ``'varname'`` if the ``variable`` is a constant, and
    ``array_name[0]`` if it is a scalar array.
    '''
    from brian2.devices.device import get_device  # avoid circular import
    if variable.array:
        return '%s[0]' % get_device().get_array_name(variable)
    else:
        return '%s' % varname


class CodeObject(Nameable):
    '''
    Executable code object.
    
    The ``code`` can either be a string or a
    `brian2.codegen.templates.MultiTemplate`.
    
    After initialisation, the code is compiled with the given namespace
    using ``code.compile(namespace)``.
    
    Calling ``code(key1=val1, key2=val2)`` executes the code with the given
    variables inserted into the namespace.
    '''
    
    #: The `CodeGenerator` class used by this `CodeObject`
    generator_class = None
    #: A short name for this type of `CodeObject`
    class_name = None

    def __init__(self, owner, code, variables, variable_indices,
                 template_name, template_source, name='codeobject*'):
        Nameable.__init__(self, name=name)
        try:    
            owner = weakref.proxy(owner)
        except TypeError:
            pass # if owner was already a weakproxy then this will be the error raised
        self.owner = owner
        self.code = code
        self.variables = variables
        self.variable_indices = variable_indices
        self.template_name = template_name
        self.template_source = template_source

    @classmethod
    def is_available(cls):
        '''
        Whether this target for code generation is available. Should use a
        minimal example to check whether code generation works in general.
        '''
        raise NotImplementedError("CodeObject class %s is missing an "
                                  "'is_available' method." % (cls.__name__))

    def update_namespace(self):
        '''
        Update the namespace for this timestep. Should only deal with variables
        where *the reference* changes every timestep, i.e. where the current
        reference in `namespace` is not correct.
        '''
        pass

    def compile(self):
        pass

    def __call__(self, **kwds):
        self.update_namespace()
        self.namespace.update(**kwds)

        return self.run()

    def run(self):
        '''
        Runs the code in the namespace.
        
        Returns
        -------
        return_value : dict
            A dictionary with the keys corresponding to the `output_variables`
            defined during the call of `CodeGenerator.code_object`.
        '''
        raise NotImplementedError()


def _error_msg(code, name):
    '''
    Little helper function for error messages.
    '''
    error_msg = 'Error generating code for code object %s ' % name
    code_lines = [l for l in code.split('\n') if len(l.strip())]
    # If the abstract code is only one line, display it in full
    if len(code_lines) <= 1:
        error_msg += 'from this abstract code: "%s"\n' % code_lines[0]
    else:
        error_msg += ('from %d lines of abstract code, first line is: '
                      '"%s"\n') % (len(code_lines), code_lines[0])
    return error_msg


def create_runner_codeobj(group, code, template_name,
                          run_namespace,
                          user_code=None,
                          variable_indices=None,
                          name=None, check_units=True,
                          needed_variables=None,
                          additional_variables=None,
                          template_kwds=None,
                          override_conditional_write=None,
                          codeobj_class=None
                          ):
    ''' Create a `CodeObject` for the execution of code in the context of a
    `Group`.

    Parameters
    ----------
    group : `Group`
        The group where the code is to be run
    code : str or dict of str
        The code to be executed.
    template_name : str
        The name of the template to use for the code.
    run_namespace : dict-like
        An additional namespace that is used for variable lookup (either
        an explicitly defined namespace or one taken from the local
        context).
    user_code : str, optional
        The code that had been specified by the user before other code was
        added automatically. If not specified, will be assumed to be identical
        to ``code``.
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
    template_kwds : dict, optional
        A dictionary of additional information that is passed to the template.
    override_conditional_write: list of str, optional
        A list of variable names which are used as conditions (e.g. for
        refractoriness) which should be ignored.
    codeobj_class : class, optional
        The `CodeObject` class to run code with. If not specified, defaults to
        the `group`'s ``codeobj_class`` attribute.
    '''

    if name is None:
        if group is not None:
            name = '%s_%s_codeobject*' % (group.name, template_name)
        else:
            name = '%s_codeobject*' % template_name

    if user_code is None:
        user_code = code

    if isinstance(code, str):
        code = {None: code}
        user_code = {None: user_code}

    msg = 'Creating code object (group=%s, template name=%s) for abstract code:\n' % (group.name, template_name)
    msg += indent(code_representation(code))
    logger.diagnostic(msg)
    from brian2.devices import get_device
    device = get_device()
    
    if override_conditional_write is None:
        override_conditional_write = set([])
    else:
        override_conditional_write = set(override_conditional_write)

    if codeobj_class is None:
        codeobj_class = device.code_object_class(group.codeobj_class)
    else:
        codeobj_class = device.code_object_class(codeobj_class)

    template = getattr(codeobj_class.templater, template_name)
    template_variables = getattr(template, 'variables', None)

    all_variables = dict(group.variables)
    if additional_variables is not None:
        all_variables.update(additional_variables)

    # Determine the identifiers that were used
    identifiers = set()
    user_identifiers = set()
    for v, u_v in zip(code.values(), user_code.values()):
        _, uk, u = analyse_identifiers(v, all_variables, recursive=True)
        identifiers |= uk | u
        _, uk, u = analyse_identifiers(u_v, all_variables, recursive=True)
        user_identifiers |= uk | u

    # Add variables that are not in the abstract code, nor specified in the
    # template but nevertheless necessary
    if needed_variables is None:
        needed_variables = []
    # Resolve all variables (variables used in the code and variables needed by
    # the template)
    variables = group.resolve_all(identifiers | set(needed_variables) | set(template_variables),
                                  # template variables are not known to the user:
                                  user_identifiers=user_identifiers,
                                  additional_variables=additional_variables,
                                  run_namespace=run_namespace)
    # We raise this error only now, because there is some non-obvious code path
    # where Jinja tries to get a Synapse's "name" attribute via syn['name'],
    # which then triggers the use of the `group_get_indices` template which does
    # not exist for standalone. Putting the check for template == None here
    # means we will first raise an error about the unknown identifier which will
    # then make Jinja try syn.name
    if template is None:
        codeobj_class_name = codeobj_class.class_name or codeobj_class.__name__
        raise AttributeError(('"%s" does not provide a code generation '
                              'template "%s"') % (codeobj_class_name,
                                                  template_name))


    conditional_write_variables = {}
    # Add all the "conditional write" variables
    for var in variables.itervalues():
        cond_write_var = getattr(var, 'conditional_write', None)
        if cond_write_var in override_conditional_write:
            continue
        if cond_write_var is not None:
            if (cond_write_var.name in variables and
                    not variables[cond_write_var.name] is cond_write_var):
                logger.diagnostic(('Variable "%s" is needed for the '
                                   'conditional write mechanism of variable '
                                   '"%s". Its name is already used for %r.') % (cond_write_var.name,
                                                                                var.name,
                                                                                variables[cond_write_var.name]))
            else:
                conditional_write_variables[cond_write_var.name] = cond_write_var

    variables.update(conditional_write_variables)

    if check_units:
        for c in code.values():
            # This is the first time that the code is parsed, catch errors
            try:
                check_units_statements(c, variables)
            except (SyntaxError, ValueError) as ex:
                error_msg = _error_msg(c, name)
                raise ValueError(error_msg + str(ex))

    all_variable_indices = copy.copy(group.variables.indices)
    if additional_variables is not None:
        all_variable_indices.update(additional_variables.indices)
    if variable_indices is not None:
        all_variable_indices.update(variable_indices)

    # Make "conditional write" variables use the same index as the variable
    # that depends on them
    for varname, var in variables.iteritems():
        cond_write_var = getattr(var, 'conditional_write', None)
        if cond_write_var is not None:
            all_variable_indices[cond_write_var.name] = all_variable_indices[varname]

    # Add the indices needed by the variables
    varnames = variables.keys()
    for varname in varnames:
        var_index = all_variable_indices[varname]
        if not var_index in ('_idx', '0'):
            variables[var_index] = all_variables[var_index]

    return device.code_object(owner=group,
                              name=name,
                              abstract_code=code,
                              variables=variables,
                              template_name=template_name,
                              variable_indices=all_variable_indices,
                              template_kwds=template_kwds,
                              codeobj_class=codeobj_class,
                              override_conditional_write=override_conditional_write,
                              )
