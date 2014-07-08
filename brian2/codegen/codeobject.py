'''
Module providing the base `CodeObject` and related functions.
'''
import copy
import weakref

from brian2.core.names import Nameable
from brian2.equations.unitcheck import check_units_statements
from brian2.utils.logger import get_logger
from brian2.utils.stringtools import indent, code_representation

from .translation import analyse_identifiers

__all__ = ['CodeObject',
           'CodeObjectUpdater',
           ]

logger = get_logger(__name__)


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

    def __init__(self, owner, code, variables, name='codeobject*'):
        Nameable.__init__(self, name=name)
        try:    
            owner = weakref.proxy(owner)
        except TypeError:
            pass # if owner was already a weakproxy then this will be the error raised
        self.owner = owner
        self.code = code
        self.variables = variables

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


def check_code_units(code, group, additional_variables=None,
                     level=0, run_namespace=None,):
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
    level : int, optional
        How far to go up in the stack to find the calling frame.
    run_namespace : dict-like, optional
        An additional namespace, as provided to `Group.before_run`

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
        resolved_namespace = group.resolve_all(unknown,
                                               level=level+1,
                                               run_namespace=run_namespace)
    except KeyError as ex:
        raise KeyError('Error occured when checking "%s": %s' % (code,
                                                                 str(ex)))

    all_variables.update(resolved_namespace)

    check_units_statements(code, all_variables)


def create_runner_codeobj(group, code, template_name,
                          variable_indices=None,
                          name=None, check_units=True,
                          needed_variables=None,
                          additional_variables=None,
                          level=0,
                          run_namespace=None,
                          template_kwds=None,
                          override_conditional_write=None,
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
    level : int, optional
        How far to go up in the stack to find the call frame.
    run_namespace : dict-like, optional
        An additional namespace that is used for variable lookup (if not
        defined, the implicit namespace of local variables is used).
    template_kwds : dict, optional
        A dictionary of additional information that is passed to the template.
    override_conditional_write: list of str, optional
        A list of variable names which are used as conditions (e.g. for
        refractoriness) which should be ignored.
    '''
    
    if isinstance(code, str):
        code = {None: code}
    msg = 'Creating code object (group=%s, template name=%s) for abstract code:\n' % (group.name, template_name)
    msg += indent(code_representation(code))
    logger.debug(msg)
    from brian2.devices import get_device
    device = get_device()
    
    if override_conditional_write is None:
        override_conditional_write = set([])
    else:
        override_conditional_write = set(override_conditional_write)

    if check_units:
        for c in code.values():
            check_code_units(c, group,
                             additional_variables=additional_variables,
                             level=level+1,
                             run_namespace=run_namespace)

    codeobj_class = device.code_object_class(group.codeobj_class)
    template = getattr(codeobj_class.templater, template_name)

    all_variables = dict(group.variables)
    if additional_variables is not None:
        all_variables.update(additional_variables)

    # Determine the identifiers that were used
    used_known = set()
    unknown = set()
    for v in code.values():
        _, uk, u = analyse_identifiers(v, all_variables, recursive=True)
        used_known |= uk
        unknown |= u

    logger.debug('Unknown identifiers in the abstract code: ' + ', '.join(unknown))

    # Only pass the variables that are actually used
    variables = group.resolve_all(used_known | unknown,
                                  additional_variables=additional_variables,
                                  run_namespace=run_namespace,
                                  level=level+1)

    conditional_write_variables = {}
    # Add all the "conditional write" variables
    for var in variables.itervalues():
        cond_write_var = getattr(var, 'conditional_write', None)
        if cond_write_var in override_conditional_write:
            continue
        if cond_write_var is not None and cond_write_var not in variables.values():
            if cond_write_var.name in variables:
                raise AssertionError(('Variable "%s" is needed for the '
                                      'conditional write mechanism of variable '
                                      '"%s". Its name is already used for %r.') % (cond_write_var.name,
                                                                                   var.name,
                                                                                   variables[cond_write_var.name]))
            conditional_write_variables[cond_write_var.name] = cond_write_var

    variables.update(conditional_write_variables)

    # Add variables that are not in the abstract code, nor specified in the
    # template but nevertheless necessary
    if needed_variables is None:
        needed_variables = []
    # Also add the variables that the template needs
    variables.update(group.resolve_all(set(needed_variables) | set(template.variables),
                                       additional_variables=additional_variables,
                                       run_namespace=run_namespace,
                                       level=level+1,
                                       do_warn=False))  # no warnings for internally used variables

    if name is None:
        if group is not None:
            name = '%s_%s_codeobject*' % (group.name, template_name)
        else:
            name = '%s_codeobject*' % template_name

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
                              codeobj_class=group.codeobj_class,
                              override_conditional_write=override_conditional_write,
                              )
