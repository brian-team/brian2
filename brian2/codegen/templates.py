"""
Handles loading templates from a directory.
"""
import re
from collections.abc import Mapping

from jinja2 import (
    Environment,
    PackageLoader,
    ChoiceLoader,
    StrictUndefined,
    TemplateNotFound,
)

from brian2.utils.stringtools import indent, strip_empty_lines, get_identifiers


__all__ = ["Templater"]

AUTOINDENT_START = "%%START_AUTOINDENT%%"
AUTOINDENT_END = "%%END_AUTOINDENT%%"


def autoindent(code):
    if isinstance(code, list):
        code = "\n".join(code)
    if not code.startswith("\n"):
        code = f"\n{code}"
    if not code.endswith("\n"):
        code = f"{code}\n"
    return AUTOINDENT_START + code + AUTOINDENT_END


def autoindent_postfilter(code):
    lines = code.split("\n")
    outlines = []
    addspaces = 0
    for line in lines:
        if AUTOINDENT_START in line:
            if addspaces > 0:
                raise SyntaxError("Cannot nest autoindents")
            addspaces = line.find(AUTOINDENT_START)
            line = line.replace(AUTOINDENT_START, "")
        if AUTOINDENT_END in line:
            line = line.replace(AUTOINDENT_END, "")
            addspaces = 0
        outlines.append(" " * addspaces + line)
    return "\n".join(outlines)


def variables_to_array_names(variables, access_data=True):
    from brian2.devices.device import get_device

    device = get_device()
    names = [device.get_array_name(var, access_data=access_data) for var in variables]
    return names


class LazyTemplateLoader(object):
    """
    Helper object to load templates only when they are needed.
    """

    def __init__(self, environment, extension):
        self.env = environment
        self.extension = extension
        self._templates = {}

    def get_template(self, name):
        if name not in self._templates:
            try:
                template = CodeObjectTemplate(
                    self.env.get_template(name + self.extension),
                    self.env.loader.get_source(self.env, name + self.extension)[0],
                )
            except TemplateNotFound:
                try:
                    # Try without extension as well (e.g. for makefiles)
                    template = CodeObjectTemplate(
                        self.env.get_template(name),
                        self.env.loader.get_source(self.env, name)[0],
                    )
                except TemplateNotFound:
                    raise KeyError(f'No template with name "{name}" found.')
            self._templates[name] = template
        return self._templates[name]


class Templater(object):
    """
    Class to load and return all the templates a `CodeObject` defines.

    Parameters
    ----------

    package_name : str, tuple of str
        The package where the templates are saved. If this is a tuple then each template will be searched in order
        starting from the first package in the tuple until the template is found. This allows for derived templates
        to be used. See also `~Templater.derive`.
    extension : str
        The file extension (e.g. ``.pyx``) used for the templates.
    env_globals : dict (optional)
        A dictionary of global values accessible by the templates. Can be used for providing utility functions.
        In all cases, the filter 'autoindent' is available (see existing templates for example usage).
    templates_dir : str, tuple of str, optional
        The name of the directory containing the templates. Defaults to ``'templates'``.

    Notes
    -----
    Templates are accessed using ``templater.template_base_name`` (the base name is without the file extension).
    This returns a `CodeObjectTemplate`.
    """

    def __init__(
        self, package_name, extension, env_globals=None, templates_dir="templates"
    ):
        if isinstance(package_name, str):
            package_name = (package_name,)
        if isinstance(templates_dir, str):
            templates_dir = (templates_dir,)
        loader = ChoiceLoader(
            [
                PackageLoader(name, t_dir)
                for name, t_dir in zip(package_name, templates_dir)
            ]
        )
        self.env = Environment(
            loader=loader,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
        )
        self.env.globals["autoindent"] = autoindent
        self.env.filters["autoindent"] = autoindent
        self.env.filters["variables_to_array_names"] = variables_to_array_names
        if env_globals is not None:
            self.env.globals.update(env_globals)
        else:
            env_globals = {}
        self.env_globals = env_globals
        self.package_names = package_name
        self.templates_dir = templates_dir
        self.extension = extension
        self.templates = LazyTemplateLoader(self.env, extension)

    def __getattr__(self, item):
        return self.templates.get_template(item)

    def derive(
        self, package_name, extension=None, env_globals=None, templates_dir="templates"
    ):
        """
        Return a new Templater derived from this one, where the new package name and globals overwrite the old.
        """
        if extension is None:
            extension = self.extension
        if isinstance(package_name, str):
            package_name = (package_name,)
        if env_globals is None:
            env_globals = {}
        if isinstance(templates_dir, str):
            templates_dir = (templates_dir,)
        package_name = package_name + self.package_names
        templates_dir = templates_dir + self.templates_dir
        new_env_globals = self.env_globals.copy()
        new_env_globals.update(**env_globals)
        return Templater(
            package_name,
            extension=extension,
            env_globals=new_env_globals,
            templates_dir=templates_dir,
        )


class CodeObjectTemplate(object):
    """
    Single template object returned by `Templater` and used for final code generation

    Should not be instantiated by the user, but only directly by `Templater`.

    Notes
    -----

    The final code is obtained from this by calling the template (see `~CodeObjectTemplater.__call__`).
    """

    def __init__(self, template, template_source):
        self.template = template
        self.template_source = template_source
        #: The set of variables in this template
        self.variables = set([])
        #: The indices over which the template iterates completely
        self.iterate_all = set([])
        #: Read-only variables that are changed by this template
        self.writes_read_only = set([])
        # This is the bit inside {} for USES_VARIABLES { list of words }
        specifier_blocks = re.findall(
            r"\bUSES_VARIABLES\b\s*\{(.*?)\}", template_source, re.M | re.S
        )
        # Same for ITERATE_ALL
        iterate_all_blocks = re.findall(
            r"\bITERATE_ALL\b\s*\{(.*?)\}", template_source, re.M | re.S
        )
        # And for WRITES_TO_READ_ONLY_VARIABLES
        writes_read_only_blocks = re.findall(
            r"\bWRITES_TO_READ_ONLY_VARIABLES\b\s*\{(.*?)\}",
            template_source,
            re.M | re.S,
        )
        #: Does this template allow writing to scalar variables?
        self.allows_scalar_write = "ALLOWS_SCALAR_WRITE" in template_source

        for block in specifier_blocks:
            self.variables.update(get_identifiers(block))
        for block in iterate_all_blocks:
            self.iterate_all.update(get_identifiers(block))
        for block in writes_read_only_blocks:
            self.writes_read_only.update(get_identifiers(block))

    def __call__(self, scalar_code, vector_code, **kwds):
        """
        Return a usable code block or blocks from this template.

        Parameters
        ----------
        scalar_code : dict
            Dictionary of scalar code blocks.
        vector_code : dict
            Dictionary of vector code blocks
        **kwds
            Additional parameters to pass to the template

        Notes
        -----

        Returns either a string (if macros were not used in the template), or a `MultiTemplate` (if macros were used).
        """
        if (
            scalar_code is not None
            and len(scalar_code) == 1
            and list(scalar_code)[0] is None
        ):
            scalar_code = scalar_code[None]
        if (
            vector_code is not None
            and len(vector_code) == 1
            and list(vector_code)[0] is None
        ):
            vector_code = vector_code[None]
        kwds["scalar_code"] = scalar_code
        kwds["vector_code"] = vector_code
        module = self.template.make_module(kwds)
        if len([k for k in module.__dict__ if not k.startswith("_")]):
            return MultiTemplate(module)
        else:
            return autoindent_postfilter(str(module))


class MultiTemplate(Mapping):
    """
    Code generated by a `CodeObjectTemplate` with multiple blocks

    Each block is a string stored as an attribute with the block name. The
    object can also be accessed as a dictionary.
    """

    def __init__(self, module):
        self._templates = {}
        for k, f in module.__dict__.items():
            if not k.startswith("_"):
                s = autoindent_postfilter(str(f()))
                setattr(self, k, s)
                self._templates[k] = s

    def __getitem__(self, item):
        return self._templates[item]

    def __iter__(self):
        return iter(self._templates)

    def __len__(self):
        return len(self._templates)

    def __str__(self):
        s = ""
        for k, v in list(self._templates.items()):
            s += f"{k}:\n"
            s += f"{strip_empty_lines(indent(v))}\n"
        return s

    __repr__ = __str__
