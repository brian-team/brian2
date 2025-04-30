"""
Module implementing the C++ "standalone" device.
"""

import inspect
import itertools
import numbers
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zlib
from collections import Counter, defaultdict
from collections.abc import Mapping
from distutils import ccompiler
from hashlib import md5

import numpy as np

from brian2.codegen.codeobject import check_compiler_kwds
from brian2.codegen.cpp_prefs import get_compiler_and_args, get_msvc_env
from brian2.codegen.generators.cpp_generator import c_data_type
from brian2.core.functions import Function
from brian2.core.namespace import get_local_namespace
from brian2.core.network import Network
from brian2.core.preferences import BrianPreference, prefs
from brian2.core.variables import (
    ArrayVariable,
    Constant,
    DynamicArrayVariable,
    Variable,
    VariableView,
)
from brian2.devices.device import Device, all_devices, reset_device, set_device
from brian2.groups.group import Group
from brian2.input import TimedArray
from brian2.parsing.rendering import CPPNodeRenderer
from brian2.synapses.synapses import Synapses
from brian2.units import second
from brian2.units.fundamentalunits import Quantity, fail_for_dimension_mismatch
from brian2.utils.filelock import FileLock
from brian2.utils.filetools import copy_directory, ensure_directory, in_directory
from brian2.utils.logger import get_logger, std_silent
from brian2.utils.stringtools import word_substitute

from .codeobject import CPPStandaloneCodeObject, openmp_pragma

__all__ = []

logger = get_logger(__name__)


# Preferences
prefs.register_preferences(
    "devices.cpp_standalone",
    "C++ standalone preferences ",
    openmp_threads=BrianPreference(
        default=0,
        docs="""
        The number of threads to use if OpenMP is turned on. By default, this value is set to 0 and the C++ code
        is generated without any reference to OpenMP. If greater than 0, then the corresponding number of threads
        are used to launch the simulation.
        """,
    ),
    openmp_spatialneuron_strategy=BrianPreference(
        default=None,
        validator=lambda val: val in [None, "branches", "systems"],
        docs="""
        DEPRECATED. Previously used to chose the strategy to parallelize the
        solution of the three tridiagonal systems for multicompartmental
        neurons. Now, its value is ignored.
        """,
    ),
    make_cmd_unix=BrianPreference(
        default="make",
        docs="""
        The make command used to compile the standalone project. Defaults to the
        standard GNU make commane "make".""",
    ),
    run_cmd_unix=BrianPreference(
        default="./main",
        validator=lambda val: isinstance(val, str) or isinstance(val, list),
        docs="""
        The command used to run the compiled standalone project. Defaults to executing
        the compiled binary with "./main". Must be a single binary as string or a list
        of command arguments (e.g. ["./binary", "--key", "value"]).
        """,
    ),
    extra_make_args_unix=BrianPreference(
        default=["-j"],
        docs="""
        Additional flags to pass to the GNU make command on Linux/OS-X.
        Defaults to "-j" for parallel compilation.""",
    ),
    extra_make_args_windows=BrianPreference(
        default=[],
        docs="""
        Additional flags to pass to the nmake command on Windows. By default, no
        additional flags are passed.
        """,
    ),
    run_environment_variables=BrianPreference(
        default={"LD_BIND_NOW": "1"},
        docs="""
        Dictionary of environment variables and their values that will be set
        during the execution of the standalone code.
        """,
    ),
)


class CPPWriter:
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.source_files = set()
        self.header_files = set()

    def write(self, filename, contents):
        logger.diagnostic(f"Writing file {filename}:\n{contents}")
        if filename.lower().endswith(".cpp") or filename.lower().endswith(".c"):
            self.source_files.add(filename)
        elif filename.lower().endswith(".h"):
            self.header_files.add(filename)
        elif filename.endswith(".*"):
            self.write(f"{filename[:-1]}cpp", contents.cpp_file)
            self.write(f"{filename[:-1]}h", contents.h_file)
            return
        fullfilename = os.path.join(self.project_dir, filename)
        if os.path.exists(fullfilename):
            with open(fullfilename) as f:
                if f.read() == contents:
                    return
        with open(fullfilename, "w") as f:
            f.write(contents)


def invert_dict(x):
    return {v: k for k, v in x.items()}


class CPPStandaloneDevice(Device):
    """
    The `Device` used for C++ standalone simulations.
    """

    def __init__(self):
        super().__init__()
        #: Dictionary mapping `ArrayVariable` objects to their globally
        #: unique name
        self.arrays = {}
        #: Dictionary mapping `ArrayVariable` objects to their value or to
        #: ``None`` if the value (potentially) depends on executed code. This
        #: mechanism allows to access state variables in standalone mode if
        #: their value is known at run time
        self.array_cache = {}
        #: List of all dynamic arrays
        #: Dictionary mapping `DynamicArrayVariable` objects with 1 dimension to
        #: their globally unique name
        self.dynamic_arrays = {}
        #: Dictionary mapping `DynamicArrayVariable` objects with 2 dimensions
        #: to their globally unique name
        self.dynamic_arrays_2d = {}
        #: List of all arrays to be filled with zeros (list of (var, varname) )
        self.zero_arrays = []
        #: List of all arrays to be filled with numbers (list of
        #: (var, varname, start) tuples
        self.arange_arrays = []
        #: Set of all existing synapses
        self.synapses = set()
        #: Whether the simulation has been run
        self.has_been_run = False

        #: Whether apply_run_args has been called
        self.run_args_applied = False

        #: Whether a run should trigger a build
        self.build_on_run = False

        #: build options
        self.build_options = None

        #: The directory which contains the generated code and results
        self.project_dir = None

        #: The directory which contains the results (relative to `project_dir``)
        self.results_dir = None

        #: Whether to generate profiling information (stored in an instance
        #: variable to be accessible during CodeObject generation)
        self.enable_profiling = False

        #: CodeObjects that use profiling (users can potentially enable
        #: profiling only for a subset of runs)
        self.profiled_codeobjects = []

        #: Dict of all static saved arrays
        self.static_arrays = {}

        #: Names of static arrays used for run_args given as lists of values
        self.run_args_arrays = []

        #: Dict of all TimedArray objects
        self.timed_arrays = {}

        self.code_objects = {}
        self.main_queue = []
        self.runfuncs = {}
        self.networks = set()
        self.static_array_specs = []
        self.report_func = ""

        #: Code lines that have been manually added with `device.insert_code`
        #: Dictionary mapping slot names to lists of lines.
        #: Note that the main slot is handled separately as part of `main_queue`
        self.code_lines = {
            "before_start": [],
            "after_start": [],
            "before_network_run": [],
            "after_network_run": [],
            "before_end": [],
            "after_end": [],
        }

        #: Dictionary storing compile and binary execution times
        self.timers = {"run_binary": None, "compile": {"clean": None, "make": None}}

        self.clocks = set()

        self.extra_compile_args = []
        self.define_macros = []
        self.headers = []
        self.include_dirs = []
        self.library_dirs = []
        self.runtime_library_dirs = []
        self.run_environment_variables = {}
        if sys.platform.startswith("darwin"):
            if "DYLD_LIBRARY_PATH" in os.environ:
                dyld_library_path = f"{os.environ['DYLD_LIBRARY_PATH']}:{os.path.join(sys.prefix, 'lib')}"
            else:
                dyld_library_path = os.path.join(sys.prefix, "lib")
            self.run_environment_variables["DYLD_LIBRARY_PATH"] = dyld_library_path
        self.libraries = []
        if sys.platform == "win32":
            self.libraries += ["advapi32"]
        self.extra_link_args = []
        self.writer = None

    def reinit(self):
        # Remember the build_on_run setting and its options -- important during
        # testing
        build_on_run = self.build_on_run
        build_options = self.build_options
        self.__init__()
        super().reinit()
        self.build_on_run = build_on_run
        self.build_options = build_options

    def spike_queue(self, source_start, source_end):
        return None  # handled differently

    def freeze(self, code, ns):
        # TODO: Remove this function at some point
        logger.warn(
            "The CPPStandaloneDevice.freeze function should no longer "
            "be used, add constant definitions directly to the "
            'code in the "CONSTANTS" section instead.',
            name_suffix="deprecated_freeze_use",
            once=True,
        )
        # this is a bit of a hack, it should be passed to the template somehow
        for k, v in ns.items():
            if isinstance(v, Variable) and v.scalar and v.constant and v.read_only:
                try:
                    v = v.get_value()
                except NotImplementedError:
                    continue
            if isinstance(v, str):
                code = word_substitute(code, {k: v})
            elif isinstance(v, numbers.Number):
                # Use a renderer to correctly transform constants such as True or inf
                renderer = CPPNodeRenderer()
                string_value = renderer.render_expr(repr(v))
                if prefs.core.default_float_dtype == np.float32 and isinstance(
                    v, (float, np.float32, np.float64)
                ):
                    string_value += "f"
                if v < 0:
                    string_value = f"({string_value})"
                code = word_substitute(code, {k: string_value})
            else:
                pass  # don't deal with this object
        return code

    def insert_code(self, slot, code):
        """
        Insert code directly into main.cpp
        """
        if slot == "main":
            self.main_queue.append(("insert_code", code))
        elif slot in self.code_lines:
            self.code_lines[slot].append(code)
        else:
            logger.warn(f"Ignoring device code, unknown slot: {slot}, code: {code}")

    def apply_run_args(self):
        if self.run_args_applied:
            raise RuntimeError(
                "The 'apply_run_args()' function can only be called once."
            )
        self.insert_code("main", "set_from_command_line(args);")
        self.run_args_applied = True

    def static_array(self, name, arr):
        arr = np.atleast_1d(arr)
        assert len(arr), f"length for {name}: {len(arr)}"
        name = f"_static_array_{name}"
        basename = name
        i = 0
        while name in self.static_arrays:
            i += 1
            name = f"{basename}_{str(i)}"
        self.static_arrays[name] = arr.copy()
        return name

    def get_array_name(self, var, access_data=True):
        """
        Return a globally unique name for `var`.

        Parameters
        ----------
        access_data : bool, optional
            For `DynamicArrayVariable` objects, specifying `True` here means the
            name for the underlying data is returned. If specifying `False`,
            the name of object itself is returned (e.g. to allow resizing).
        """
        if isinstance(var, DynamicArrayVariable):
            if access_data:
                return self.arrays[var]
            elif var.ndim == 1:
                return self.dynamic_arrays[var]
            else:
                return self.dynamic_arrays_2d[var]
        elif isinstance(var, ArrayVariable):
            return self.arrays[var]
        else:
            raise TypeError(f"Do not have a name for variable of type {type(var)}.")

    def get_array_filename(self, var, basedir=None):
        """
        Return a file name for a variable.

        Parameters
        ----------
        var : `ArrayVariable`
            The variable to get a filename for.
        basedir : str
            The base directory for the filename, defaults to ``'results'``.
            DEPRECATED: Will raise an error if specified.
        Returns
        -------
        filename : str
            A filename of the form
            ``varname+'_'+str(zlib.crc32(varname))``, where varname
            is the name returned by `get_array_name`.

        Notes
        -----
        The reason that the filename is not simply ``varname`` is
        that this could lead to file names that are not unique in file systems
        that are not case sensitive (e.g. on Windows).
        """
        if basedir is not None:
            raise ValueError("Specifying 'basedir' is no longer supported.")
        varname = self.get_array_name(var, access_data=False)
        return f"{varname}_{str(zlib.crc32(varname.encode('utf-8')))}"

    def add_array(self, var):
        # Note that a dynamic array variable is added to both the arrays and
        # the _dynamic_array dictionary
        if isinstance(var, DynamicArrayVariable):
            # The code below is slightly more complicated than just looking
            # for a unique name as above for static_array, the name has
            # potentially to be unique for more than one dictionary, with
            # different prefixes. This is because dynamic arrays are added to
            # a ``dynamic_arrays`` dictionary (with a `_dynamic` prefix) and to
            # the general ``arrays`` dictionary. We want to make sure that we
            # use the same name in the two dictionaries, not for example
            # ``_dynamic_array_source_name_2`` and ``_array_source_name_1``
            # (this would work fine, but it would make the code harder to read).
            orig_dynamic_name = dynamic_name = (
                f"_dynamic_array_{var.owner.name}_{var.name}"
            )
            orig_array_name = array_name = f"_array_{var.owner.name}_{var.name}"
            suffix = 0

            if var.ndim == 1:
                dynamic_dict = self.dynamic_arrays
            elif var.ndim == 2:
                dynamic_dict = self.dynamic_arrays_2d
            else:
                raise AssertionError(
                    "Did not expect a dynamic array with {var.ndim} dimensions."
                )
            while (
                dynamic_name in dynamic_dict.values()
                or array_name in self.arrays.values()
            ):
                suffix += 1
                dynamic_name = f"{orig_dynamic_name}_{int(suffix)}"
                array_name = f"{orig_array_name}_{int(suffix)}"
            dynamic_dict[var] = dynamic_name
            self.arrays[var] = array_name
        else:
            orig_array_name = array_name = f"_array_{var.owner.name}_{var.name}"
            suffix = 0
            while array_name in self.arrays.values():
                suffix += 1
                array_name = f"{orig_array_name}_{int(suffix)}"
            self.arrays[var] = array_name

    def init_with_zeros(self, var, dtype):
        if isinstance(var, DynamicArrayVariable):
            varname = f"_dynamic{self.arrays[var]}"
        else:
            varname = self.arrays[var]
        self.zero_arrays.append((var, varname))
        self.array_cache[var] = np.zeros(var.size, dtype=dtype)

    def init_with_arange(self, var, start, dtype):
        if isinstance(var, DynamicArrayVariable):
            varname = f"_dynamic{self.arrays[var]}"
        else:
            varname = self.arrays[var]
        self.arange_arrays.append((var, varname, start))
        self.array_cache[var] = np.arange(0, var.size, dtype=dtype) + start

    def fill_with_array(self, var, arr):
        arr = np.asarray(arr)
        if arr.size == 0:
            return  # nothing to do
        array_name = self.get_array_name(var, access_data=False)
        if isinstance(var, DynamicArrayVariable):
            # We can never be sure about the size of a dynamic array, so
            # we can't do correct broadcasting. Therefore, we do not cache
            # them at all for now.
            self.array_cache[var] = None
        else:
            new_arr = np.empty(var.size, dtype=var.dtype)
            new_arr[:] = arr
            self.array_cache[var] = new_arr

        if arr.size == 1:
            if var.size == 1:
                value = CPPNodeRenderer().render_expr(repr(arr.item(0)))
                # For a single assignment, generate a code line instead of storing the array
                self.main_queue.append(("set_by_single_value", (array_name, 0, value)))
            else:
                self.main_queue.append(
                    (
                        "set_by_constant",
                        (array_name, arr.item(), isinstance(var, DynamicArrayVariable)),
                    )
                )
        else:
            # Using the std::vector instead of a pointer to the underlying
            # data for dynamic arrays is fast enough here and it saves us some
            # additional work to set up the pointer
            static_array_name = self.static_array(array_name, arr)
            self.main_queue.append(
                (
                    "set_by_array",
                    (
                        array_name,
                        static_array_name,
                        isinstance(var, DynamicArrayVariable),
                    ),
                )
            )

    def resize(self, var, new_size):
        array_name = self.get_array_name(var, access_data=False)
        self.main_queue.append(("resize_array", (array_name, new_size)))

    def variableview_set_with_index_array(self, variableview, item, value, check_units):
        if isinstance(item, slice) and item == slice(None):
            item = "True"
        value = Quantity(value)

        if (
            isinstance(item, int) or (isinstance(item, np.ndarray) and item.shape == ())
        ) and value.size == 1:
            array_name = self.get_array_name(variableview.variable, access_data=False)
            value_str = CPPNodeRenderer().render_expr(repr(np.asarray(value).item(0)))
            if self.array_cache.get(variableview.variable, None) is not None:
                self.array_cache[variableview.variable][item] = value
            # For a single assignment, generate a code line instead of storing the array
            self.main_queue.append(
                ("set_by_single_value", (array_name, item, value_str))
            )
        # Simple case where we don't have to do any indexing
        elif item == "True" and variableview.index_var in ("_idx", "0"):
            self.fill_with_array(variableview.variable, value)
        else:
            # We have to calculate indices. This will not work for synaptic
            # variables
            try:
                indices = np.asarray(
                    variableview.indexing(item, index_var=variableview.index_var)
                )
            except NotImplementedError:
                raise NotImplementedError(
                    f"Cannot set variable '{variableview.name}' "
                    "this way in standalone, try using "
                    "string expressions."
                )
            # Using the std::vector instead of a pointer to the underlying
            # data for dynamic arrays is fast enough here and it saves us some
            # additional work to set up the pointer
            arrayname = self.get_array_name(variableview.variable, access_data=False)
            if indices.shape != () and (
                value.shape == () or (value.size == 1 and indices.size > 1)
            ):
                value = np.repeat(value, indices.size)
            elif value.shape != indices.shape and len(value) != len(indices):
                raise ValueError(
                    "Provided values do not match the size "
                    "of the indices, "
                    f"{len(value)} != len(indices)."
                )

            staticarrayname_index = self.static_array(f"_index_{arrayname}", indices)
            staticarrayname_value = self.static_array(f"_value_{arrayname}", value)
            self.array_cache[variableview.variable] = None
            self.main_queue.append(
                (
                    "set_array_by_array",
                    (arrayname, staticarrayname_index, staticarrayname_value),
                )
            )

    def get_value(self, var, access_data=True):
        # Usually, we cannot retrieve the values of state variables in
        # standalone scripts since their values might depend on the evaluation
        # of expressions at runtime. For some variables we do know the value
        # however (values that have been set with explicit values and not
        # changed in code objects)
        if self.array_cache.get(var, None) is not None:
            return self.array_cache[var]
        else:
            # After the network has been run, we can retrieve the values from
            # disk
            if self.has_been_run:
                dtype = var.dtype
                fname = os.path.join(self.results_dir, self.get_array_filename(var))
                with open(fname, "rb") as f:
                    data = np.fromfile(f, dtype=dtype)
                # This is a bit of an heuristic, but our 2d dynamic arrays are
                # only expanding in one dimension, we assume here that the
                # other dimension has size 0 at the beginning
                if isinstance(var.size, tuple) and len(var.size) == 2:
                    if var.size[0] * var.size[1] == len(data):
                        size = var.size
                    elif var.size[0] == 0:
                        size = (len(data) // var.size[1], var.size[1])
                    elif var.size[1] == 0:
                        size = (var.size[0], len(data) // var.size[0])
                    else:
                        raise IndexError(
                            "Do not now how to deal with 2d "
                            f"array of size {var.size!s}, the array on "
                            f"disk has length {len(data)}."
                        )

                    var.size = size
                    return data.reshape(var.size)
                var.size = len(data)
                return data
            raise NotImplementedError(
                "Cannot retrieve the values of state "
                "variables in standalone code before the "
                "simulation has been run."
            )

    def variableview_get_subexpression_with_index_array(
        self, variableview, item, run_namespace=None
    ):
        if not self.has_been_run:
            raise NotImplementedError(
                "Cannot retrieve the values of state "
                "variables in standalone code before the "
                "simulation has been run."
            )
        # Temporarily switch to the runtime device to evaluate the subexpression
        # (based on the values stored on disk)
        set_device("runtime")
        result = VariableView.get_subexpression_with_index_array(
            variableview, item, run_namespace=run_namespace
        )
        reset_device()
        return result

    def variableview_get_with_expression(self, variableview, code, run_namespace=None):
        raise NotImplementedError(
            "Cannot retrieve the values of state "
            "variables with string expressions in "
            "standalone scripts."
        )

    def code_object_class(self, codeobj_class=None, fallback_pref=None):
        """
        Return `CodeObject` class (either `CPPStandaloneCodeObject` class or input)

        Parameters
        ----------
        codeobj_class : a `CodeObject` class, optional
            If this is keyword is set to None or no arguments are given, this method will return
            the default (`CPPStandaloneCodeObject` class).
        fallback_pref : str, optional
            For the cpp_standalone device this option is ignored.

        Returns
        -------
        codeobj_class : class
            The `CodeObject` class that should be used
        """
        # Ignore the requested pref (used for optimization in runtime)
        if codeobj_class is None:
            return CPPStandaloneCodeObject
        else:
            return codeobj_class

    def code_object(
        self,
        owner,
        name,
        abstract_code,
        variables,
        template_name,
        variable_indices,
        codeobj_class=None,
        template_kwds=None,
        override_conditional_write=None,
        compiler_kwds=None,
    ):
        if compiler_kwds is None:
            compiler_kwds = {}
        check_compiler_kwds(
            compiler_kwds,
            [
                "headers",
                "sources",
                "define_macros",
                "libraries",
                "include_dirs",
                "library_dirs",
                "runtime_library_dirs",
            ],
            "C++ standalone",
        )
        if template_kwds is None:
            template_kwds = dict()
        else:
            template_kwds = dict(template_kwds)
        # In standalone mode, the only place where we use additional header
        # files is by inserting them into the template
        codeobj_headers = compiler_kwds.get("headers", [])
        template_kwds["user_headers"] = (
            self.headers + prefs["codegen.cpp.headers"] + codeobj_headers
        )
        template_kwds["profiled"] = self.enable_profiling

        do_not_invalidate = set()
        if template_name == "synapses_create_array":
            cache = self.array_cache
            if (
                cache[variables["N"]] is None
            ):  # synapses have been previously created with code
                # Nothing we can do
                logger.debug(
                    f"Synapses for '{owner.name}' have previously been created with "
                    "code, we therefore cannot cache the synapses created with arrays "
                    f"via '{name}'",
                    name_suffix="code_created_synapses_exist",
                )
            else:  # first time we create synapses, or all previous connect calls were with arrays
                cache[variables["N"]][0] += variables["sources"].size
                do_not_invalidate.add(variables["N"])
                for var, value in [
                    (
                        variables["_synaptic_pre"],
                        variables["sources"].get_value()
                        + variables["_source_offset"].get_value(),
                    ),
                    (
                        variables["_synaptic_post"],
                        variables["targets"].get_value()
                        + variables["_target_offset"].get_value(),
                    ),
                ]:
                    cache[var] = np.append(
                        cache.get(var, np.empty(0, dtype=int)), value
                    )
                    do_not_invalidate.add(var)

        codeobj = super().code_object(
            owner,
            name,
            abstract_code,
            variables,
            template_name,
            variable_indices,
            codeobj_class=codeobj_class,
            template_kwds=template_kwds,
            override_conditional_write=override_conditional_write,
            compiler_kwds=compiler_kwds,
        )
        self.code_objects[codeobj.name] = codeobj
        if self.enable_profiling:
            self.profiled_codeobjects.append(codeobj.name)

        for var in codeobj.variables.values():
            if isinstance(var, TimedArray):
                self.timed_arrays[var] = var.name
        # We mark all writeable (i.e. not read-only) variables used by the code
        # as "dirty" to avoid that the cache contains incorrect values. This
        # might remove a number of variables from the cache unnecessarily,
        # since many variables are only read in the code.
        # On the other hand, there are also *read-only* variables that can be
        # changed by code (the "read-only" attribute only refers to the user
        # being able to change values directly). For example, synapse creation
        # write source and target indices, and monitors write the shared values.
        # To correctly mark these values as changed, templates can include a
        # "WRITES_TO_READ_ONLY_VARIABLES" comment, stating the name of the
        # changed variables. For a monitor, this would for example state that
        # the number of recorded values "N" changes. For the recorded variables,
        # however, this information cannot be included in the template because
        # it is up to the user to define which variables are recorded. For such
        # cases, the "owner" object (e.g. a SpikeMonitor) can define a
        # "written_readonly_vars" attribute, storing a set of `Variable` objects
        # that will be changed by the owner's code objects.
        template = getattr(codeobj.templater, template_name)
        written_readonly_vars = {
            codeobj.variables[varname] for varname in template.writes_read_only
        } | getattr(owner, "written_readonly_vars", set())
        for var in codeobj.variables.values():
            if (
                isinstance(var, ArrayVariable)
                and var not in do_not_invalidate
                and (not var.read_only or var in written_readonly_vars)
            ):
                self.array_cache[var] = None

        return codeobj

    def check_openmp_compatible(self, nb_threads):
        if nb_threads > 0:
            logger.warn(
                "OpenMP code is not yet well tested, and may be inaccurate.",
                "openmp",
                once=True,
            )
            logger.diagnostic(f"Using OpenMP with {int(nb_threads)} threads ")
            if prefs.devices.cpp_standalone.openmp_spatialneuron_strategy is not None:
                logger.warn(
                    "The devices.cpp_standalone.openmp_spatialneuron_strategy "
                    "preference is no longer used and will be removed in "
                    "future versions of Brian.",
                    "openmp_spatialneuron_strategy",
                    once=True,
                )

    def generate_objects_source(
        self,
        writer,
        arange_arrays,
        synapses,
        static_array_specs,
        networks,
        timed_arrays,
    ):
        arr_tmp = self.code_object_class().templater.objects(
            None,
            None,
            array_specs=self.arrays,
            dynamic_array_specs=self.dynamic_arrays,
            dynamic_array_2d_specs=self.dynamic_arrays_2d,
            zero_arrays=self.zero_arrays,
            arange_arrays=arange_arrays,
            synapses=synapses,
            clocks=self.clocks,
            static_array_specs=static_array_specs,
            networks=networks,
            get_array_filename=self.get_array_filename,
            get_array_name=self.get_array_name,
            profiled_codeobjects=self.profiled_codeobjects,
            code_objects=list(self.code_objects.values()),
            timed_arrays=timed_arrays,
        )
        writer.write("objects.*", arr_tmp)

    def generate_main_source(self, writer):
        main_lines = []
        procedures = [("", main_lines)]
        runfuncs = {}
        for func, args in self.main_queue:
            if func == "before_run_code_object":
                (codeobj,) = args
                main_lines.append(f"_before_run_{codeobj.name}();")
            elif func == "run_code_object":
                (codeobj,) = args
                main_lines.append(f"_run_{codeobj.name}();")
            elif func == "after_run_code_object":
                (codeobj,) = args
                main_lines.append(f"_after_run_{codeobj.name}();")
            elif func == "run_network":
                net, netcode = args
                main_lines.extend(netcode)
            elif func == "set_by_constant":
                arrayname, value, is_dynamic = args
                size_str = f"{arrayname}.size()" if is_dynamic else f"_num_{arrayname}"
                code = f"""
                {openmp_pragma('static')}
                for(int i=0; i<{size_str}; i++)
                {{
                    {arrayname}[i] = {CPPNodeRenderer().render_expr(repr(value))};
                }}
                """
                main_lines.extend(code.split("\n"))
            elif func == "set_by_array":
                arrayname, staticarrayname, is_dynamic = args
                size_str = f"{arrayname}.size()" if is_dynamic else f"_num_{arrayname}"
                code = f"""
                {openmp_pragma('static')}
                for(int i=0; i<{size_str}; i++)
                {{
                    {arrayname}[i] = {staticarrayname}[i];
                }}
                """
                main_lines.extend(code.split("\n"))
            elif func == "set_by_single_value":
                arrayname, item, value = args
                code = f"{arrayname}[{item}] = {value};"
                main_lines.extend([code])
            elif func == "set_array_by_array":
                arrayname, staticarrayname_index, staticarrayname_value = args
                code = f"""
                {openmp_pragma('static')}
                for(int i=0; i<_num_{staticarrayname_index}; i++)
                {{
                    {arrayname}[{staticarrayname_index}[i]] = {staticarrayname_value}[i];
                }}
                """
                main_lines.extend(code.split("\n"))
            elif func == "resize_array":
                array_name, new_size = args
                main_lines.append(f"{array_name}.resize({new_size});")
            elif func == "insert_code":
                main_lines.append(args)
            elif func == "start_run_func":
                name, include_in_parent = args
                if include_in_parent:
                    main_lines.append(f"{name}();")
                main_lines = []
                procedures.append((name, main_lines))
            elif func == "end_run_func":
                name, include_in_parent = args
                name, main_lines = procedures.pop(-1)
                runfuncs[name] = main_lines
                name, main_lines = procedures[-1]
            elif func == "seed":
                seed = args
                nb_threads = prefs.devices.cpp_standalone.openmp_threads
                if nb_threads == 0:  # no OpenMP
                    nb_threads = 1
                main_lines.append(f"for (int _i=0; _i<{nb_threads}; _i++)")
                if seed is None:  # random
                    main_lines.append("    brian::_random_generators[_i].seed();")
                else:
                    main_lines.append(
                        f"    brian::_random_generators[_i].seed({seed!r}L + _i);"
                    )
            else:
                raise NotImplementedError(f"Unknown main queue function type {func}")

        self.runfuncs = runfuncs

        # generate the finalisations
        for codeobj in self.code_objects.values():
            if hasattr(codeobj.code, "main_finalise"):
                main_lines.append(codeobj.code.main_finalise)

        user_headers = self.headers + prefs["codegen.cpp.headers"]
        main_tmp = self.code_object_class().templater.main(
            None,
            None,
            main_lines=main_lines,
            code_lines=self.code_lines,
            code_objects=list(self.code_objects.values()),
            report_func=self.report_func,
            dt=float(self.defaultclock.dt),
            user_headers=user_headers,
        )
        writer.write("main.cpp", main_tmp)

    def generate_codeobj_source(self, writer):
        # Generate data for non-constant values
        renderer = CPPNodeRenderer()
        code_object_defs = defaultdict(list)
        for codeobj in self.code_objects.values():
            lines = []
            for k, v in codeobj.variables.items():
                if isinstance(v, ArrayVariable):
                    try:
                        if isinstance(v, DynamicArrayVariable):
                            if v.ndim == 1:
                                dyn_array_name = self.dynamic_arrays[v]
                                array_name = self.arrays[v]
                                c_type = c_data_type(v.dtype)
                                line = (
                                    f"{c_type}* const {array_name} ="
                                    f" {dyn_array_name}.empty()? 0 :"
                                    f" &{dyn_array_name}[0];"
                                )
                                lines.append(line)
                                line = (
                                    f"const size_t _num{k} = {dyn_array_name}.size();"
                                )
                                lines.append(line)
                        else:
                            lines.append(f"const size_t _num{k} = {v.size};")
                    except TypeError:
                        pass
                elif isinstance(v, Constant):
                    value = renderer.render_expr(repr(v.value))
                    c_type = c_data_type(v.dtype)
                    line = f"const {c_type} {k} = {value};"
                    lines.append(line)
            for line in lines:
                # Sometimes an array is referred to by to different keys in our
                # dictionary -- make sure to never add a line twice
                if line not in code_object_defs[codeobj.name]:
                    code_object_defs[codeobj.name].append(line)

        # Generate the code objects
        for codeobj in self.code_objects.values():
            # Before/after run code
            for block in codeobj.before_after_blocks:
                cpp_code = getattr(codeobj.code, f"{block}_cpp_file")
                cpp_code = cpp_code.replace(
                    "%CONSTANTS%", "\n".join(code_object_defs[codeobj.name])
                )
                h_code = getattr(codeobj.code, f"{block}_h_file")
                writer.write(f"code_objects/{block}_{codeobj.name}.cpp", cpp_code)
                writer.write(f"code_objects/{block}_{codeobj.name}.h", h_code)

            # Main code
            code = codeobj.code.cpp_file
            code = code.replace(
                "%CONSTANTS%", "\n".join(code_object_defs[codeobj.name])
            )

            writer.write(f"code_objects/{codeobj.name}.cpp", code)
            writer.write(f"code_objects/{codeobj.name}.h", codeobj.code.h_file)

    def generate_network_source(self, writer, compiler):
        maximum_run_time = self._maximum_run_time
        if maximum_run_time is not None:
            maximum_run_time = float(maximum_run_time)
        network_tmp = self.code_object_class().templater.network(
            None, None, maximum_run_time=maximum_run_time
        )
        writer.write("network.*", network_tmp)

    def generate_synapses_classes_source(self, writer):
        synapses_classes_tmp = self.code_object_class().templater.synapses_classes(
            None, None
        )
        writer.write("synapses_classes.*", synapses_classes_tmp)

    def generate_run_source(self, writer):
        run_tmp = self.code_object_class().templater.run(
            None,
            None,
            run_funcs=self.runfuncs,
            code_objects=list(self.code_objects.values()),
            user_headers=self.headers,
            array_specs=self.arrays,
            clocks=self.clocks,
        )
        writer.write("run.*", run_tmp)

    def generate_makefile(
        self, writer, compiler, compiler_flags, linker_flags, nb_threads, debug
    ):
        if compiler == "msvc":
            if nb_threads > 1:
                openmp_flag = "/openmp"
            else:
                openmp_flag = ""
            if debug:
                compiler_debug_flags = "/DEBUG /DDEBUG"
                linker_debug_flags = "/DEBUG"
            else:
                compiler_debug_flags = ""
                linker_debug_flags = ""
            # Generate the visual studio makefile
            source_bases = [
                fname.replace(".cpp", "").replace(".c", "").replace("/", "\\")
                for fname in sorted(writer.source_files)
            ]
            win_makefile_tmp = self.code_object_class().templater.win_makefile(
                None,
                None,
                source_files=sorted(writer.source_files),
                source_bases=source_bases,
                compiler_flags=compiler_flags,
                compiler_debug_flags=compiler_debug_flags,
                linker_flags=linker_flags,
                linker_debug_flags=linker_debug_flags,
                openmp_flag=openmp_flag,
            )
            writer.write("win_makefile", win_makefile_tmp)
            # write the list of sources
            source_list = " ".join(source_bases)
            source_list_fname = os.path.join(self.project_dir, "sourcefiles.txt")
            if os.path.exists(source_list_fname):
                with open(source_list_fname) as f:
                    if f.read() == source_list:
                        return
            with open(source_list_fname, "w") as f:
                f.write(source_list)
        else:
            # Generate the makefile
            if os.name == "nt":
                rm_cmd = "del *.o /s\n\tdel main.exe $(DEPS)"
            else:
                rm_cmd = "rm $(OBJS) $(PROGRAM) $(DEPS)"
            if debug:
                compiler_debug_flags = "-g -DDEBUG"
                linker_debug_flags = "-g"
            else:
                compiler_debug_flags = ""
                linker_debug_flags = ""
            makefile_tmp = self.code_object_class().templater.makefile(
                None,
                None,
                source_files=" ".join(sorted(writer.source_files)),
                header_files=" ".join(sorted(writer.header_files)),
                compiler_flags=compiler_flags,
                compiler_debug_flags=compiler_debug_flags,
                linker_debug_flags=linker_debug_flags,
                linker_flags=linker_flags,
                rm_cmd=rm_cmd,
            )
            writer.write("makefile", makefile_tmp)

    def copy_source_files(self, writer, directory):
        # Copy the brianlibdirectory
        brianlib_dir = os.path.join(
            os.path.split(inspect.getsourcefile(CPPStandaloneCodeObject))[0], "brianlib"
        )
        brianlib_files = copy_directory(
            brianlib_dir, os.path.join(directory, "brianlib")
        )
        for file in brianlib_files:
            if file.lower().endswith(".cpp"):
                writer.source_files.add(f"brianlib/{file}")
            elif file.lower().endswith(".h"):
                writer.header_files.add(f"brianlib/{file}")

        # Copy the CSpikeQueue implementation
        shutil.copy2(
            os.path.join(
                os.path.split(inspect.getsourcefile(Synapses))[0], "cspikequeue.cpp"
            ),
            os.path.join(directory, "brianlib", "spikequeue.h"),
        )
        shutil.copy2(
            os.path.join(
                os.path.split(inspect.getsourcefile(Synapses))[0], "stdint_compat.h"
            ),
            os.path.join(directory, "brianlib", "stdint_compat.h"),
        )

    def _insert_func_namespace(self, func, code_object, namespace):
        impl = func.implementations[self.code_object_class()]
        func_namespace = impl.get_namespace(code_object.owner)
        if func_namespace is not None:
            namespace.update(func_namespace)
        if impl.dependencies is not None:
            for dep in impl.dependencies.values():
                self._insert_func_namespace(dep, code_object, namespace)

    def write_static_arrays(self, directory):
        # Write Function namespaces as static arrays
        for code_object in self.code_objects.values():
            for var in code_object.variables.values():
                if isinstance(var, Function):
                    self._insert_func_namespace(var, code_object, self.static_arrays)

        logger.diagnostic(f"static arrays: {str(sorted(self.static_arrays.keys()))}")

        static_array_specs = []
        for name, arr in sorted(self.static_arrays.items()):
            arr.tofile(os.path.join(directory, "static_arrays", name))
            static_array_specs.append((name, c_data_type(arr.dtype), arr.size, name))
        self.static_array_specs = static_array_specs

    def compile_source(self, directory, compiler, debug, clean):
        with in_directory(directory):
            if compiler == "msvc":
                msvc_env, vcvars_cmd = get_msvc_env()
                make_cmd = "nmake /f win_makefile"
                make_args = " ".join(
                    prefs.devices.cpp_standalone.extra_make_args_windows
                )
                if os.path.exists("winmake.log"):
                    os.remove("winmake.log")
                if vcvars_cmd:
                    with open("winmake.log", "w") as f:
                        f.write(f"{vcvars_cmd}\n")
                else:
                    with open("winmake.log", "w") as f:
                        f.write("MSVC environment: \n")
                        for key, value in msvc_env.items():
                            f.write(f"{key}={value}\n")
                with std_silent(debug):
                    if vcvars_cmd:
                        if clean:
                            start_time = time.time()
                            os.system(
                                f"{vcvars_cmd} >>winmake.log 2>&1 && {make_cmd} clean >"
                                " NUL 2>&1"
                            )
                            self.timers["compile"]["clean"] = time.time() - start_time
                        start_time = time.time()
                        x = os.system(
                            f"{vcvars_cmd} >>winmake.log 2>&1 &&"
                            f" {make_cmd} {make_args}>>winmake.log 2>&1"
                        )
                        self.timers["compile"]["make"] = time.time() - start_time
                    else:
                        os.environ.update(msvc_env)
                        if clean:
                            start_time = time.time()
                            os.system(f"{make_cmd} clean > NUL 2>&1")
                            self.timers["compile"]["clean"] = time.time() - start_time
                        start_time = time.time()
                        x = os.system(f"{make_cmd} {make_args}>>winmake.log 2>&1")
                        self.timers["compile"]["make"] = time.time() - start_time

                    if x != 0:
                        error_message = f"Project compilation failed (error code: {x}), consider having a look at 'winmake.log'."
                        if not clean:
                            error_message += (
                                " Consider running with "
                                '"clean=True" to force a complete '
                                "rebuild."
                            )
                        raise RuntimeError(error_message)
            else:
                with std_silent(debug):
                    if clean:
                        start_time = time.time()
                        os.system("make clean >/dev/null 2>&1")
                        self.timers["compile"]["clean"] = time.time() - start_time
                    make_cmd = prefs.devices.cpp_standalone.make_cmd_unix
                    make_args = " ".join(
                        prefs.devices.cpp_standalone.extra_make_args_unix
                    )
                    start_time = time.time()
                    x = os.system(f"{make_cmd} {make_args}")
                    self.timers["compile"]["make"] = time.time() - start_time
                    if x != 0:
                        error_message = (
                            "Project compilation failed (error code: %u)." % x
                        )
                        if not clean:
                            error_message += (
                                " Consider running with "
                                '"clean=True" to force a complete '
                                "rebuild."
                            )
                        raise RuntimeError(error_message)

    def seed(self, seed=None):
        """
        Set the seed for the random number generator.

        Parameters
        ----------
        seed : int, optional
            The seed value for the random number generator, or ``None`` (the
            default) to set a random seed.
        """
        self.main_queue.append(("seed", seed))

    def run(
        self, directory=None, results_directory=None, with_output=True, run_args=None
    ):
        if directory is None:
            directory = self.project_dir
        if results_directory is None:
            results_directory = self.results_dir
        else:
            if os.path.isabs(results_directory):
                raise TypeError(
                    "The 'results_directory' argument needs to be a relative path but"
                    f" was '{results_directory}'."
                )
            # Translate path to absolute path which ends with /
            self.results_dir = os.path.join(
                os.path.abspath(os.path.join(directory, results_directory)), ""
            )
        ensure_directory(self.results_dir)

        self.run_args_arrays.clear()  # forget about arrays from previous runs
        if run_args is None:
            run_args = []
        elif isinstance(run_args, Mapping):
            list_rep = []
            for key, value in run_args.items():
                if isinstance(key, VariableView):
                    (
                        name,
                        string_value,
                        value_name,
                        value_ar,
                    ) = self._prepare_variableview_run_arg(key, value)
                elif isinstance(key, TimedArray):
                    (
                        name,
                        string_value,
                        value_name,
                        value_ar,
                    ) = self._prepare_timed_array_run_arg(key, value)
                else:
                    raise TypeError(
                        "The keys for 'run_args' need to be 'VariableView' objects,"
                        " i.e. attributes of groups such as 'neurongroup.v', or a"
                        f" 'TimedArray'. Key has type '{type(key)}' instead."
                    )
                if value_name:
                    fname = os.path.join(self.project_dir, "static_arrays", value_name)
                    # Make sure processes trying to write the same file don't clash
                    with FileLock(fname + ".lock"):
                        if not os.path.exists(fname):
                            value_ar.tofile(fname)
                    self.run_args_arrays.append(value_name)
                list_rep.append(f"{name}={string_value}")

            run_args = list_rep

        # Invalidate array cache for all variables set on the command line
        for arg in run_args:
            s = arg.split("=")
            if len(s) == 2:
                for var in self.array_cache:
                    if (
                        hasattr(var.owner, "name")
                        and var.owner.name + "." + var.name == s[0]
                    ):
                        self.array_cache[var] = None
        run_args = ["--results_dir", self.results_dir] + run_args
        # Invalidate the cached end time of the clock and network, to deal with stopped simulations
        for clock in self.clocks:
            self.array_cache[clock.variables["t"]] = None

        with in_directory(directory):
            # Set environment variables

            for key, value in itertools.chain(
                prefs["devices.cpp_standalone.run_environment_variables"].items(),
                self.run_environment_variables.items(),
            ):
                if key in os.environ and os.environ[key] != value:
                    logger.info(
                        f'Overwriting environment variable "{key}"',
                        name_suffix="overwritten_env_var",
                        once=True,
                    )
                os.environ[key] = value
            if not with_output:
                stdout = open(os.path.join(self.results_dir, "stdout.txt"), "w")
            else:
                stdout = None
            if os.name == "nt":
                start_time = time.time()
                Network._globally_running = True
                x = subprocess.call(["main"] + run_args, stdout=stdout)
                self.timers["run_binary"] = time.time() - start_time
                Network._globally_running = False
            else:
                run_cmd = prefs.devices.cpp_standalone.run_cmd_unix
                if isinstance(run_cmd, str):
                    run_cmd = [run_cmd]
                start_time = time.time()
                Network._globally_running = True
                x = subprocess.call(run_cmd + run_args, stdout=stdout)
                self.timers["run_binary"] = time.time() - start_time
                Network._globally_running = False
            if stdout is not None:
                stdout.close()
            if x:
                stdout_fname = os.path.join(self.results_dir, "stdout.txt")
                if os.path.exists(stdout_fname):
                    with open(stdout_fname) as f:
                        print(f.read())
                raise RuntimeError(
                    "Project run failed (project directory:"
                    f" {os.path.abspath(directory)})"
                )
            self.has_been_run = True
            run_info_fname = os.path.join(self.results_dir, "last_run_info.txt")
            if os.path.isfile(run_info_fname):
                with open(run_info_fname) as f:
                    last_run_info = f.read()
                run_time, completed_fraction = last_run_info.split()
                self._last_run_time = float(run_time)
                self._last_run_completed_fraction = float(completed_fraction)

        # Make sure that integration did not create NaN or very large values
        owners = [var.owner for var in self.arrays]
        # We don't want to check the same owner twice but var.owner is a
        # weakproxy which we can't put into a set. We therefore store the name
        # of all objects we already checked. Furthermore, under some specific
        # instances a variable might have been created whose owner no longer
        # exists (e.g. a `_sub_idx` variable for a subgroup) -- we ignore the
        # resulting reference error.
        already_checked = set()
        for owner in owners:
            try:
                if not hasattr(owner, "name") or owner.name in already_checked:
                    continue
                if isinstance(owner, Group):
                    owner._check_for_invalid_states()
                    already_checked.add(owner.name)
            except ReferenceError:
                pass

    def _prepare_variableview_run_arg(self, key, value):
        fail_for_dimension_mismatch(key.dim, value)  # TODO: Give name of variable
        value_ar = np.asarray(value, dtype=key.dtype)
        if value_ar.ndim == 0 or value_ar.size == 1:
            # single value, give value directly on command line
            string_value = repr(value_ar.item())
            value_name = None
        else:
            if value_ar.ndim != 1 or (
                not key.variable.dynamic and value_ar.size != key.shape[0]
            ):
                raise TypeError(
                    "Incorrect size for variable"
                    f" '{key.group_name}.{key.name}'. Shape {key.shape} ≠"
                    f" {value_ar.shape}."
                )
            value_name = (
                f"init_{key.group_name}_{key.name}_{md5(value_ar.data).hexdigest()}.dat"
            )
            string_value = os.path.join("static_arrays", value_name)
        name = f"{key.group_name}.{key.name}"
        return name, string_value, value_name, value_ar

    def _prepare_timed_array_run_arg(self, key, value):
        fail_for_dimension_mismatch(key.dim, value)  # TODO: Give name of variable
        value_ar = np.asarray(value, dtype=key.values.dtype)
        if value_ar.ndim == 0 or value_ar.size == 1:
            # single value, give value directly on command line
            string_value = repr(value_ar.item())
            value_name = None
        elif value_ar.shape == key.values.shape:
            value_name = f"init_{key.name}_values_{md5(value_ar.data).hexdigest()}.dat"
            string_value = os.path.join("static_arrays", value_name)
        else:
            raise TypeError(
                "Incorrect size for variable"
                f" '{key.name}.values'. Shape {key.values.shape} ≠"
                f" {value_ar.shape}."
            )
        name = f"{key.name}.values"
        return name, string_value, value_name, value_ar

    def build(
        self,
        directory="output",
        results_directory="results",
        compile=True,
        run=True,
        debug=False,
        clean=False,
        with_output=True,
        additional_source_files=None,
        run_args=None,
        direct_call=True,
        **kwds,
    ):
        """
        Build the project

        TODO: more details

        Parameters
        ----------
        directory : str, optional
            The output directory to write the project to, any existing files
            will be overwritten. If the given directory name is ``None``, then
            a temporary directory will be used (used in the test suite to avoid
            problems when running several tests in parallel). Defaults to
            ``'output'``.
        compile : bool, optional
            Whether or not to attempt to compile the project. Defaults to
            ``True``.
        run : bool, optional
            Whether or not to attempt to run the built project if it
            successfully builds. Defaults to ``True``.
        debug : bool, optional
            Whether to compile in debug mode. Defaults to ``False``.
        with_output : bool, optional
            Whether or not to show the ``stdout`` of the built program when run.
            Output will be shown in case of compilation or runtime error.
            Defaults to ``True``.
        clean : bool, optional
            Whether or not to clean the project before building. Defaults to
            ``False``.
        additional_source_files : list of str, optional
            A list of additional ``.cpp`` files to include in the build.
        direct_call : bool, optional
            Whether this function was called directly. Is used internally to
            distinguish an automatic build due to the ``build_on_run`` option
            from a manual ``device.build`` call.
        """
        if self.build_on_run and direct_call:
            raise RuntimeError(
                "You used set_device with build_on_run=True "
                "(the default option), which will automatically "
                "build the simulation at the first encountered "
                "run call - do not call device.build manually "
                "in this case. If you want to call it manually, "
                "e.g. because you have multiple run calls, use "
                "set_device with build_on_run=False."
            )
        if self.has_been_run:
            raise RuntimeError(
                "The network has already been built and run "
                "before. To build several simulations in "
                'the same script, call "device.reinit()" '
                'and "device.activate()". Note that you '
                "will have to set build options (e.g. the "
                "directory) and defaultclock.dt again."
            )
        renames = {
            "project_dir": "directory",
            "compile_project": "compile",
            "run_project": "run",
        }
        if len(kwds):
            msg = ""
            for kwd in kwds:
                if kwd in renames:
                    msg += (
                        f"Keyword argument '{kwd}' has been renamed to "
                        f"'{renames[kwd]}'. "
                    )
                else:
                    msg += f"Unknown keyword argument '{kwd}'. "
            raise TypeError(msg)

        if additional_source_files is None:
            additional_source_files = []
        if run_args is None:
            run_args = []
        if directory is None:
            directory = tempfile.mkdtemp(prefix="brian_standalone_")
        self.project_dir = directory
        ensure_directory(directory)
        if os.path.isabs(results_directory):
            raise TypeError(
                "The 'results_directory' argument needs to be a relative path but was "
                f"'{results_directory}'."
            )
        # Translate path to absolute path which ends with /
        self.results_dir = os.path.join(
            os.path.abspath(os.path.join(directory, results_directory)), ""
        )

        # Determine compiler flags and directories
        compiler, default_extra_compile_args = get_compiler_and_args()
        extra_compile_args = self.extra_compile_args + default_extra_compile_args
        extra_link_args = self.extra_link_args + prefs["codegen.cpp.extra_link_args"]

        codeobj_define_macros = [
            macro
            for codeobj in self.code_objects.values()
            for macro in codeobj.compiler_kwds.get("define_macros", [])
        ]
        define_macros = (
            self.define_macros
            + prefs["codegen.cpp.define_macros"]
            + codeobj_define_macros
        )

        codeobj_include_dirs = [
            include_dir
            for codeobj in self.code_objects.values()
            for include_dir in codeobj.compiler_kwds.get("include_dirs", [])
        ]
        include_dirs = (
            self.include_dirs + prefs["codegen.cpp.include_dirs"] + codeobj_include_dirs
        )

        codeobj_library_dirs = [
            library_dir
            for codeobj in self.code_objects.values()
            for library_dir in codeobj.compiler_kwds.get("library_dirs", [])
        ]
        library_dirs = (
            self.library_dirs + prefs["codegen.cpp.library_dirs"] + codeobj_library_dirs
        )

        codeobj_runtime_dirs = [
            runtime_dir
            for codeobj in self.code_objects.values()
            for runtime_dir in codeobj.compiler_kwds.get("runtime_library_dirs", [])
        ]
        runtime_library_dirs = (
            self.runtime_library_dirs
            + prefs["codegen.cpp.runtime_library_dirs"]
            + codeobj_runtime_dirs
        )

        codeobj_libraries = [
            library
            for codeobj in self.code_objects.values()
            for library in codeobj.compiler_kwds.get("libraries", [])
        ]
        libraries = self.libraries + prefs["codegen.cpp.libraries"] + codeobj_libraries

        compiler_obj = ccompiler.new_compiler(compiler=compiler)

        # Distutils does not use the shell, so it does not need to quote filenames/paths
        # Since we include the compiler flags in the makefile, we need to quote them
        include_dirs = [f'"{include_dir}"' for include_dir in include_dirs]
        library_dirs = [f'"{library_dir}"' for library_dir in library_dirs]
        runtime_library_dirs = [
            f'"{runtime_dir}"' for runtime_dir in runtime_library_dirs
        ]

        compiler_flags = (
            ccompiler.gen_preprocess_options(define_macros, include_dirs)
            + extra_compile_args
        )

        linker_flags = (
            ccompiler.gen_lib_options(
                compiler_obj,
                library_dirs=library_dirs,
                runtime_library_dirs=runtime_library_dirs,
                libraries=libraries,
            )
            + extra_link_args
        )

        codeobj_source_files = [
            source_file
            for codeobj in self.code_objects.values()
            for source_file in codeobj.compiler_kwds.get("sources", [])
        ]
        additional_source_files += codeobj_source_files

        for d in ["code_objects", "results", "static_arrays"]:
            ensure_directory(os.path.join(directory, d))

        self.writer = CPPWriter(directory)

        # Get the number of threads if specified in an openmp context
        nb_threads = prefs.devices.cpp_standalone.openmp_threads
        # If the number is negative, we need to throw an error
        if nb_threads < 0:
            raise ValueError("The number of OpenMP threads can not be negative !")

        logger.diagnostic(
            "Writing C++ standalone project to directory "
            f"'{os.path.normpath(directory)}'."
        )

        self.check_openmp_compatible(nb_threads)

        self.write_static_arrays(directory)

        # Check that all names are globally unique
        names = [obj.name for net in self.networks for obj in net.sorted_objects]
        non_unique_names = [name for name, count in Counter(names).items() if count > 1]
        if len(non_unique_names):
            formatted_names = ", ".join(f"'{name}'" for name in non_unique_names)
            raise ValueError(
                "All objects need to have unique names in "
                "standalone mode, the following name(s) were used "
                f"more than once: {formatted_names}"
            )

        self.generate_objects_source(
            self.writer,
            self.arange_arrays,
            self.synapses,
            self.static_array_specs,
            self.networks,
            self.timed_arrays,
        )
        self.generate_main_source(self.writer)
        self.generate_codeobj_source(self.writer)
        self.generate_network_source(self.writer, compiler)
        self.generate_synapses_classes_source(self.writer)
        self.generate_run_source(self.writer)
        self.copy_source_files(self.writer, directory)

        self.writer.source_files.update(additional_source_files)

        self.generate_makefile(
            self.writer,
            compiler,
            compiler_flags=" ".join(compiler_flags),
            linker_flags=" ".join(linker_flags),
            nb_threads=nb_threads,
            debug=debug,
        )

        if compile:
            self.compile_source(directory, compiler, debug, clean)
            if run:
                self.run(directory, results_directory, with_output, run_args)
        time_measurements = {
            "'make clean'": self.timers["compile"]["clean"],
            "'make'": self.timers["compile"]["make"],
            "running 'main'": self.timers["run_binary"],
        }
        logged_times = [
            f"{task}: {measurement:.2f}s"
            for task, measurement in time_measurements.items()
            if measurement is not None
        ]
        logger.debug(f"Time measurements: {', '.join(logged_times)}")

    def delete(self, code=True, data=True, run_args=True, directory=True, force=False):
        if self.project_dir is None:
            return  # Nothing to delete

        if directory and not all([code, data, run_args]):
            raise ValueError(
                "When deleting the directory, code and data will"
                "be deleted as well. Set the corresponding "
                "parameters to True."
            )

        fnames = []

        # Delete data
        if data:
            results_dir = self.results_dir
            logger.debug(f"Deleting data files in '{results_dir}'")
            fnames.append(os.path.join(results_dir, "last_run_info.txt"))
            if self.profiled_codeobjects:
                fnames.append(os.path.join(results_dir, "profiling_info.txt"))
            for var in self.arrays:
                fnames.append(os.path.join(results_dir, self.get_array_filename(var)))

        # Delete code
        if code:
            logger.debug(f"Deleting code files in '{self.project_dir}'")
            if sys.platform == "win32":
                fnames.extend(
                    [
                        "sourcefiles.txt",
                        "win_makefile",
                        "main.exe",
                        "main.ilk",
                        "main.pdb",
                        "winmake.log",
                    ]
                )
            else:
                fnames.extend(["make.deps", "makefile", "main"])

            fnames.extend(
                [
                    os.path.join("brianlib", "spikequeue.h"),
                    os.path.join("brianlib", "stdint_compat.h"),
                ]
            )
            fnames.extend(self.writer.header_files)

            for source_file in self.writer.source_files:
                fnames.append(source_file)
                base_name, _ = os.path.splitext(source_file)
                if sys.platform == "win32":
                    fnames.append(f"{base_name}.obj")
                else:
                    fnames.append(f"{base_name}.o")

            for static_array_name in self.static_arrays:
                fnames.append(os.path.join("static_arrays", static_array_name))

        if run_args:
            for fname in self.run_args_arrays:
                fnames.append(os.path.join("static_arrays", fname))

        for fname in fnames:
            full_fname = os.path.join(self.project_dir, fname)
            try:
                os.remove(full_fname)
            except OSError as ex:
                logger.warn(f'File "{full_fname}" could not be deleted: {str(ex)}')

        # Delete directories

        if directory:
            directories = [
                "brianlib",
                "code_objects",
                "results",
                "static_arrays",
                "",
            ]
            full_directories = [
                os.path.join(self.project_dir, directory) for directory in directories
            ]
            for full_directory in full_directories:
                try:
                    os.rmdir(full_directory)
                except OSError:
                    if not os.path.exists(full_directory):
                        continue

                    # The directory is not empty:
                    if force:
                        logger.debug(
                            'Directory "{}" is not empty, but '
                            "deleting it due to the use of the force "
                            "option."
                        )
                        shutil.rmtree(full_directory)
                    else:
                        # We only give a warning if there is a file or a
                        # directory we do not know about. We do not want to e.g.
                        # complain about an unknown file in the results
                        # directory and then again complain about the results
                        # directory when deleting the main directory
                        still_present = [
                            name
                            for name in os.listdir(full_directory)
                            if os.path.isfile(name)
                            or os.path.join(full_directory, name)
                            not in full_directories
                        ]
                        if len(still_present):
                            still_present = ", ".join(
                                f'"{name}"' for name in still_present
                            )
                            logger.warn(
                                f"Not deleting the '{full_directory}' directory, "
                                "because it contains files/directories "
                                f"not added by Brian: {still_present}",
                                name_suffix="delete_skips_directory",
                            )

    def network_run(
        self,
        net,
        duration,
        report=None,
        report_period=10 * second,
        namespace=None,
        profile=None,
        level=0,
        **kwds,
    ):
        if duration < 0:
            raise ValueError(
                f"Function 'run' expected a non-negative duration but got '{duration}'"
            )

        self.networks.add(net)
        if kwds:
            logger.warn(
                "Unsupported keyword argument(s) provided for run: %s"
                % ", ".join(kwds.keys())
            )
        # We store this as an instance variable for later access by the
        # `code_object` method
        self.enable_profiling = profile

        # Allow setting `profile` in the `set_device` call (used e.g. in brian2cuda
        # SpeedTest configurations)
        if profile is None:
            self.enable_profiling = self.build_options.get("profile", False)

        all_objects = net.sorted_objects
        net._clocks = {obj.clock for obj in all_objects}
        t_end = net.t + duration
        for clock in net._clocks:
            clock.set_interval(net.t, t_end)

        # Get the local namespace
        if namespace is None:
            namespace = get_local_namespace(level=level + 2)

        net.before_run(namespace)
        self.synapses |= {s for s in net.objects if isinstance(s, Synapses)}
        self.clocks.update(net._clocks)
        net.t_ = float(t_end)

        # TODO: remove this horrible hack
        for clock in self.clocks:
            if clock.name == "clock":
                clock._name = "_clock"

        # Extract all the CodeObjects
        # Note that since we ran the Network object, these CodeObjects will be sorted into the right
        # running order, assuming that there is only one clock
        code_objects = []
        for obj in all_objects:
            if obj.active:
                for codeobj in obj._code_objects:
                    code_objects.append((obj.clock, codeobj))

        # Code for a progress reporting function
        standard_code = """
        std::string _format_time(float time_in_s)
        {
            float divisors[] = {24*60*60, 60*60, 60, 1};
            char letters[] = {'d', 'h', 'm', 's'};
            float remaining = time_in_s;
            std::string text = "";
            int time_to_represent;
            for (int i =0; i < sizeof(divisors)/sizeof(float); i++)
            {
                time_to_represent = int(remaining / divisors[i]);
                remaining -= time_to_represent * divisors[i];
                if (time_to_represent > 0 || text.length())
                {
                    if(text.length() > 0)
                    {
                        text += " ";
                    }
                    text += (std::to_string(time_to_represent)+letters[i]);
                }
            }
            //less than one second
            if(text.length() == 0)
            {
                text = "< 1s";
            }
            return text;
        }
        void report_progress(const double elapsed, const double completed, const double start, const double duration)
        {
            if (completed == 0.0)
            {
                %STREAMNAME% << "Starting simulation at t=" << start << " s for duration " << duration << " s";
            } else
            {
                %STREAMNAME% << completed*duration << " s (" << (int)(completed*100.) << "%) simulated in " << _format_time(elapsed);
                if (completed < 1.0)
                {
                    const int remaining = (int)((1-completed)/completed*elapsed+0.5);
                    %STREAMNAME% << ", estimated " << _format_time(remaining) << " remaining.";
                }
            }

            %STREAMNAME% << std::endl << std::flush;
        }
        """
        if report is None:
            report_func = ""
        elif report == "text" or report == "stdout":
            report_func = standard_code.replace("%STREAMNAME%", "std::cout")
        elif report == "stderr":
            report_func = standard_code.replace("%STREAMNAME%", "std::cerr")
        elif isinstance(report, str):
            report_func = """
            void report_progress(const double elapsed, const double completed, const double start, const double duration)
            {
            %REPORT%
            }
            """.replace(
                "%REPORT%", report
            )
        else:
            raise TypeError(
                "report argument has to be either 'text', "
                "'stdout', 'stderr', or the code for a report "
                "function"
            )

        if report_func != "":
            if self.report_func != "" and report_func != self.report_func:
                raise NotImplementedError(
                    "The C++ standalone device does not "
                    "support multiple report functions, "
                    "each run has to use the same (or "
                    "none)."
                )
            self.report_func = report_func

        if report is not None:
            report_call = "report_progress"
        else:
            report_call = "NULL"

        # Generate the updaters
        run_lines = [f"{net.name}.clear();"]
        all_clocks = set()
        for clock, codeobj in code_objects:
            run_lines.append(f"{net.name}.add(&{clock.name}, _run_{codeobj.name});")
            all_clocks.add(clock)

        # Under some rare circumstances (e.g. a NeuronGroup only defining a
        # subexpression that is used by other groups (via linking, or recorded
        # by a StateMonitor) *and* not calculating anything itself *and* using a
        # different clock than all other objects) a clock that is not used by
        # any code object should nevertheless advance during the run. We include
        # such clocks without a code function in the network.
        for clock in net._clocks:
            if clock not in all_clocks:
                run_lines.append(f"{net.name}.add(&{clock.name}, NULL);")

        run_lines.extend(self.code_lines["before_network_run"])
        if not self.run_args_applied:
            run_lines.append("set_from_command_line(args);")
            self.run_args_applied = True
        run_lines.append(
            f"{net.name}.run({float(duration)!r}, {report_call},"
            f" {float(report_period)!r});"
        )
        run_lines.extend(self.code_lines["after_network_run"])
        self.main_queue.append(("run_network", (net, run_lines)))

        net.after_run()

        # Manually set the cache for the clocks, simulation scripts might
        # want to access the time (which has been set in code and is therefore
        # not accessible by the normal means until the code has been built and
        # run)
        for clock in net._clocks:
            self.array_cache[clock.variables["timestep"]] = np.array([clock._i_end])
            self.array_cache[clock.variables["t"]] = np.array(
                [clock._i_end * clock.dt_]
            )

        if self.build_on_run:
            if self.has_been_run:
                raise RuntimeError(
                    "The network has already been built and run "
                    "before. Use set_device with "
                    "build_on_run=False and an explicit "
                    "device.build call to use multiple run "
                    "statements with this device."
                )
            self.build(direct_call=False, **self.build_options)

    def network_store(self, net, *args, **kwds):
        raise NotImplementedError(
            "The store/restore mechanism is not supported in the C++ standalone"
        )

    def network_restore(self, net, *args, **kwds):
        raise NotImplementedError(
            "The store/restore mechanism is not supported in the C++ standalone"
        )

    def network_get_profiling_info(self, net):
        fname = os.path.join(self.project_dir, "results", "profiling_info.txt")
        if not os.path.exists(fname):
            raise ValueError(
                "No profiling info collected (did you run with 'profile=True'?)"
            )
        net._profiling_info = []
        with open(fname) as f:
            for line in f:
                (key, val) = line.split()
                net._profiling_info.append((key, float(val) * second))
        return sorted(net._profiling_info, key=lambda item: item[1], reverse=True)

    def run_function(self, name, include_in_parent=True):
        """
        Context manager to divert code into a function

        Code that happens within the scope of this context manager will go into the named function.

        Parameters
        ----------
        name : str
            The name of the function to divert code into.
        include_in_parent : bool
            Whether or not to include a call to the newly defined function in the parent context.
        """
        return RunFunctionContext(name, include_in_parent)


class RunFunctionContext:
    def __init__(self, name, include_in_parent):
        self.name = name
        self.include_in_parent = include_in_parent

    def __enter__(self):
        cpp_standalone_device.main_queue.append(
            ("start_run_func", (self.name, self.include_in_parent))
        )

    def __exit__(self, type, value, traceback):
        cpp_standalone_device.main_queue.append(
            ("end_run_func", (self.name, self.include_in_parent))
        )


cpp_standalone_device = CPPStandaloneDevice()
all_devices["cpp_standalone"] = cpp_standalone_device
