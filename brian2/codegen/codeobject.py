"""
Module providing the base `CodeObject` and related functions.
"""

import collections
import copy
import platform

from brian2.core.base import weakproxy_with_fallback
from brian2.core.functions import DEFAULT_FUNCTIONS, Function
from brian2.core.names import Nameable
from brian2.equations.unitcheck import check_units_statements
from brian2.utils.logger import get_logger
from brian2.utils.stringtools import code_representation, indent

from .translation import analyse_identifiers

__all__ = ["CodeObject", "constant_or_scalar"]

logger = get_logger(__name__)


#: Dictionary with basic information about the current system (OS, etc.)
sys_info = {
    "system": platform.system(),
    "architecture": platform.architecture(),
    "machine": platform.machine(),
}


def constant_or_scalar(varname, variable):
    """
    Convenience function to generate code to access the value of a variable.
    Will return ``'varname'`` if the ``variable`` is a constant, and
    ``array_name[0]`` if it is a scalar array.
    """
    from brian2.devices.device import get_device  # avoid circular import

    if variable.array:
        return f"{get_device().get_array_name(variable)}[0]"
    else:
        return f"{varname}"


class CodeObject(Nameable):
    """
    Executable code object.

    The ``code`` can either be a string or a
    `brian2.codegen.templates.MultiTemplate`.

    After initialisation, the code is compiled with the given namespace
    using ``code.compile(namespace)``.

    Calling ``code(key1=val1, key2=val2)`` executes the code with the given
    variables inserted into the namespace.
    """

    #: The `CodeGenerator` class used by this `CodeObject`
    generator_class = None
    #: A short name for this type of `CodeObject`
    class_name = None

    def __init__(
        self,
        owner,
        code,
        variables,
        variable_indices,
        template_name,
        template_source,
        compiler_kwds,
        name="codeobject*",
    ):
        Nameable.__init__(self, name=name)
        self.owner = weakproxy_with_fallback(owner)
        self.code = code
        self.compiled_code = {}
        self.variables = variables
        self.variable_indices = variable_indices
        self.template_name = template_name
        self.template_source = template_source
        self.compiler_kwds = compiler_kwds

    def __getstate__(self):
        state = self.__dict__.copy()
        state["owner"] = self.owner.__repr__.__self__
        # Replace Function objects for standard functions by their name
        state["variables"] = self.variables.copy()
        for k, v in state["variables"].items():
            if isinstance(v, Function) and v is DEFAULT_FUNCTIONS[k]:
                state["variables"][k] = k
        return state

    def __setstate__(self, state):
        state["owner"] = weakproxy_with_fallback(state["owner"])
        for k, v in state["variables"].items():
            if isinstance(v, str):
                state["variables"][k] = DEFAULT_FUNCTIONS[k]
        self.__dict__ = state

    @classmethod
    def is_available(cls):
        """
        Whether this target for code generation is available. Should use a
        minimal example to check whether code generation works in general.
        """
        raise NotImplementedError(
            f"CodeObject class {cls.__name__} is missing an 'is_available' method."
        )

    def update_namespace(self):
        """
        Update the namespace for this timestep. Should only deal with variables
        where *the reference* changes every timestep, i.e. where the current
        reference in `namespace` is not correct.
        """
        pass

    def compile_block(self, block):
        raise NotImplementedError("Implement compile_block method")

    def compile(self):
        for block in ["before_run", "run", "after_run"]:
            self.compiled_code[block] = self.compile_block(block)

    def __call__(self, **kwds):
        self.update_namespace()
        self.namespace.update(**kwds)

        return self.run()

    def run_block(self, block):
        raise NotImplementedError("Implement run_block method")

    def before_run(self):
        """
        Runs the preparation code in the namespace. This code will only be
        executed once per run.

        Returns
        -------
        return_value : dict
            A dictionary with the keys corresponding to the `output_variables`
            defined during the call of `CodeGenerator.code_object`.
        """
        return self.run_block("before_run")

    def run(self):
        """
        Runs the main code in the namespace.

        Returns
        -------
        return_value : dict
            A dictionary with the keys corresponding to the `output_variables`
            defined during the call of `CodeGenerator.code_object`.
        """
        return self.run_block("run")

    def after_run(self):
        """
        Runs the finalizing code in the namespace. This code will only be
        executed once per run.

        Returns
        -------
        return_value : dict
            A dictionary with the keys corresponding to the `output_variables`
            defined during the call of `CodeGenerator.code_object`.
        """
        return self.run_block("after_run")


def _error_msg(code, name):
    """
    Little helper function for error messages.
    """
    error_msg = f"Error generating code for code object '{name}' "
    code_lines = [line for line in code.split("\n") if len(line.strip())]
    # If the abstract code is only one line, display it in full
    if len(code_lines) <= 1:
        error_msg += f"from this abstract code: '{code_lines[0]}'\n"
    else:
        error_msg += (
            f"from {len(code_lines)} lines of abstract code, first line is: "
            "'code_lines[0]'\n"
        )
    return error_msg


def check_compiler_kwds(compiler_kwds, accepted_kwds, target):
    """
    Internal function to check the provided compiler keywords against the list
    of understood keywords.

    Parameters
    ----------
    compiler_kwds : dict
        Dictionary of compiler keywords and respective list of values.
    accepted_kwds : list of str
        The compiler keywords understood by the code generation target
    target : str
        The name of the code generation target (used for the error message).

    Raises
    ------
    ValueError
        If a compiler keyword is not understood
    """
    for key in compiler_kwds:
        if key not in accepted_kwds:
            formatted_kwds = ", ".join(f"'{kw}'" for kw in accepted_kwds)
            raise ValueError(
                f"The keyword argument '{key}' is not understood by "
                f"the code generation target '{target}'. The valid "
                f"arguments are: {formatted_kwds}."
            )


def _merge_compiler_kwds(list_of_kwds):
    """
    Merges a list of keyword dictionaries. Values in these dictionaries are
    lists of values, the merged dictionaries will contain the concatenations
    of lists specified for the same key.

    Parameters
    ----------
    list_of_kwds : list of dict
        A list of compiler keyword dictionaries that should be merged.

    Returns
    -------
    merged_kwds : dict
        The merged dictionary
    """
    merged_kwds = collections.defaultdict(list)
    for kwds in list_of_kwds:
        for key, values in kwds.items():
            if not isinstance(values, list):
                raise TypeError(
                    f"Compiler keyword argument '{key}' requires a list of values."
                )
            merged_kwds[key].extend(values)
    return merged_kwds


def _gather_compiler_kwds(function, codeobj_class):
    """
    Gather all the compiler keywords for a function and its dependencies.

    Parameters
    ----------
    function : `Function`
        The function for which the compiler keywords should be gathered
    codeobj_class : type
        The class of `CodeObject` to use

    Returns
    -------
    kwds : dict
        A dictionary with the compiler arguments, a list of values for each
        key.
    """
    implementation = function.implementations[codeobj_class]
    all_kwds = [implementation.compiler_kwds]
    if implementation.dependencies is not None:
        for dependency in implementation.dependencies.values():
            all_kwds.append(_gather_compiler_kwds(dependency, codeobj_class))
    return _merge_compiler_kwds(all_kwds)


def create_runner_codeobj(
    group,
    code,
    template_name,
    run_namespace,
    user_code=None,
    variable_indices=None,
    name=None,
    check_units=True,
    needed_variables=None,
    additional_variables=None,
    template_kwds=None,
    override_conditional_write=None,
    codeobj_class=None,
):
    """Create a `CodeObject` for the execution of code in the context of a
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
    """

    if name is None:
        if group is not None:
            name = f"{group.name}_{template_name}_codeobject*"
        else:
            name = f"{template_name}_codeobject*"

    if user_code is None:
        user_code = code

    if isinstance(code, str):
        code = {None: code}
        user_code = {None: user_code}

    msg = (
        f"Creating code object (group={group.name}, template name={template_name}) for"
        " abstract code:\n"
    )
    msg += indent(code_representation(code))
    logger.diagnostic(msg)
    from brian2.devices import get_device

    device = get_device()

    if override_conditional_write is None:
        override_conditional_write = set()
    else:
        override_conditional_write = set(override_conditional_write)

    if codeobj_class is None:
        codeobj_class = device.code_object_class(group.codeobj_class)
    else:
        codeobj_class = device.code_object_class(codeobj_class)

    template = getattr(codeobj_class.templater, template_name)
    template_variables = getattr(template, "variables", None)

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
    variables = group.resolve_all(
        sorted(identifiers | set(needed_variables) | set(template_variables)),
        # template variables are not known to the user:
        user_identifiers=user_identifiers,
        additional_variables=additional_variables,
        run_namespace=run_namespace,
    )
    # We raise this error only now, because there is some non-obvious code path
    # where Jinja tries to get a Synapse's "name" attribute via syn['name'],
    # which then triggers the use of the `group_get_indices` template which does
    # not exist for standalone. Putting the check for template == None here
    # means we will first raise an error about the unknown identifier which will
    # then make Jinja try syn.name
    if template is None:
        codeobj_class_name = codeobj_class.class_name or codeobj_class.__name__
        raise AttributeError(
            f"'{codeobj_class_name}' does not provide a code "
            f"generation template '{template_name}'"
        )

    conditional_write_variables = {}
    # Add all the "conditional write" variables
    for var in variables.values():
        cond_write_var = getattr(var, "conditional_write", None)
        if cond_write_var in override_conditional_write:
            continue
        if cond_write_var is not None:
            if (
                cond_write_var.name in variables
                and not variables[cond_write_var.name] is cond_write_var
            ):
                logger.diagnostic(
                    f"Variable '{cond_write_var.name}' is needed for the "
                    "conditional write mechanism of variable "
                    f"'{var.name}'. Its name is already used for "
                    f"{variables[cond_write_var.name]!r}."
                )
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
                raise ValueError(error_msg) from ex

    all_variable_indices = copy.copy(group.variables.indices)
    if additional_variables is not None:
        all_variable_indices.update(additional_variables.indices)
    if variable_indices is not None:
        all_variable_indices.update(variable_indices)

    # Make "conditional write" variables use the same index as the variable
    # that depends on them
    for varname, var in variables.items():
        cond_write_var = getattr(var, "conditional_write", None)
        if cond_write_var is not None:
            all_variable_indices[cond_write_var.name] = all_variable_indices[varname]

    # Check that all functions are available
    for varname, value in variables.items():
        if isinstance(value, Function):
            try:
                value.implementations[codeobj_class]
            except KeyError as ex:
                # if we are dealing with numpy, add the default implementation
                from brian2.codegen.runtime.numpy_rt import NumpyCodeObject

                if codeobj_class is NumpyCodeObject:
                    value.implementations.add_numpy_implementation(value.pyfunc)
                else:
                    raise NotImplementedError(
                        f"Cannot use function '{varname}': {ex}"
                    ) from ex

    # Gather the additional compiler arguments declared by function
    # implementations
    all_keywords = [
        _gather_compiler_kwds(var, codeobj_class)
        for var in variables.values()
        if isinstance(var, Function)
    ]
    compiler_kwds = _merge_compiler_kwds(all_keywords)

    # Add the indices needed by the variables
    for varname in list(variables):
        var_index = all_variable_indices[varname]
        if var_index not in ("_idx", "0"):
            variables[var_index] = all_variables[var_index]

    return device.code_object(
        owner=group,
        name=name,
        abstract_code=code,
        variables=variables,
        template_name=template_name,
        variable_indices=all_variable_indices,
        template_kwds=template_kwds,
        codeobj_class=codeobj_class,
        override_conditional_write=override_conditional_write,
        compiler_kwds=compiler_kwds,
    )
