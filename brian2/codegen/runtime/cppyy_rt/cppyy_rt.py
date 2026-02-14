"""
cppyy runtime code object for Brian2.

Each code block (before_run, run, after_run) becomes a C++ function JIT-compiled
by cppyy/Cling. Functions receive all data as typed parameters — numpy arrays get
passed as raw pointers with zero-copy via cppyy's buffer protocol.

Three naming worlds need to stay in sync:
  1. RuntimeDevice:  "_array_neurongroup_v"
  2. C++ params:     "_ptr_array_neurongroup_v"
  3. C++ body:       "_ptr_array_neurongroup_v[_idx]"

(2) and (3) match automatically. We bridge (1)→(2) in variables_to_namespace().
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from brian2.core.base import BrianObjectException
from brian2.core.functions import Function
from brian2.core.preferences import BrianPreference, prefs
from brian2.core.variables import (
    ArrayVariable,
    AuxiliaryVariable,
    Constant,
    DynamicArrayVariable,
    Subexpression,
    Variable,
)
from brian2.utils.logger import get_logger

from ...codeobject import check_compiler_kwds
from ...generators.cpp_generator import c_data_type
from ...generators.cppyy_generator import CppyyCodeGenerator
from ...targets import codegen_targets
from ...templates import Templater
from ..numpy_rt import NumpyCodeObject

__all__: list[str] = ["CppyyCodeObject"]

logger = get_logger(__name__)

# --- Type aliases ---
# (cpp_param_name, namespace_key, c_type_string)
ParamTuple = tuple[str, str, str]
# (namespace_key, callable that returns current value)
NonconstantEntry = tuple[str, Callable[[], Any]]

# --- Preferences ---
prefs.register_preferences(
    "codegen.runtime.cppyy",
    "cppyy runtime codegen preferences",
    extra_compile_args=BrianPreference(
        default=[],
        docs="Extra flags passed to cppyy/Cling, e.g. ['-O2', '-ffast-math'].",
    ),
    enable_introspection=BrianPreference(
        default=False,
        docs="""
        Enable runtime introspection of compiled C++ code.

        When True, all compiled code objects register with a global introspector
        that allows viewing generated C++ source, parameter mappings, namespace
        contents, and even replacing functions at runtime.

        Adds minor overhead (stores source strings, maintains registry), so
        leave disabled for production runs.

        Usage:
            prefs.codegen.runtime.cppyy.enable_introspection = True
            from brian2.codegen.runtime.cppyy_rt.introspector import get_introspector
            intro = get_introspector()
        """,
    ),
)

# --- Lazy cppyy import ---
_cppyy: Any = None


def _get_cppyy() -> Any:
    """Import cppyy on first use so we don't blow up at import time."""
    global _cppyy
    if _cppyy is None:
        try:
            import cppyy

            _cppyy = cppyy
        except ImportError:
            raise ImportError(
                "cppyy is required for the cppyy runtime target. "
                "Install it with: pip install cppyy"
            ) from None
    return _cppyy


def _cppyy_c_data_type(dtype: type | np.dtype) -> str:
    """
    Like c_data_type but maps bool→int8_t instead of char.

    cppyy enforces strict type matching on buffers: numpy's bool_ viewed as
    int8 needs int8_t in the signature, not char (which is a distinct type in C++).
    """
    ctype: str = c_data_type(dtype)
    if ctype == "char":
        return "int8_t"
    return ctype


# --- One-time support code init ---
_support_code_initialized: bool = False


def _ensure_support_code() -> None:
    """
    Define universal C++ helpers exactly once in cppyy's interpreter.

    Covers: The DynamicArray header from brianlib, standard headers, Brian2's _brian_mod/_brian_pow/etc., int_(),
    and the shared MT19937 RNG engine. Guarded so repeated calls are no-ops.
    """
    global _support_code_initialized
    if _support_code_initialized:
        return

    cppyy = _get_cppyy()

    # Add brianlib include path and load dynamic_array.h ──
    # This makes DynamicArray1D<T> and DynamicArray2D<T> available to
    # all subsequently compiled cppyy code. These are the SAME classes
    # that the Cython DynamicArray wrappers use, so pointers are
    # compatible across the two FFI boundaries.
    import brian2

    brianlib_path = os.path.join(
        os.path.dirname(brian2.__file__), "devices", "cpp_standalone", "brianlib"
    )
    cppyy.add_include_path(brianlib_path)

    # Include the header — Cling compiles it and knows the class layout.
    # After this, cppyy C++ code can use DynamicArray1D<double>*, etc.
    cppyy.include("dynamic_array.h")
    from brian2.codegen.generators.cpp_generator import _universal_support_code

    guarded_code: str = f"""
    #ifndef _BRIAN2_CPPYY_SUPPORT_CODE
    #define _BRIAN2_CPPYY_SUPPORT_CODE

    #include <cmath>
    #include <cstdint>
    #include <cstdlib>
    #include <algorithm>
    #include <random>
    #include <limits>

    #ifndef M_PI
    #define M_PI 3.14159265358979323846
    #endif

    #ifndef INFINITY
    #define INFINITY (std::numeric_limits<double>::infinity())
    #endif

    // Brian2 universal support code: type promotion, _brian_mod, _brian_floordiv, etc.
    {_universal_support_code}

    // int_() — standalone gets this from stdint_compat.h, we define it here
    template<typename T>
    inline int32_t int_(T value) {{ return static_cast<int32_t>(value); }}

    // Shared RNG for rand/randn/poisson
    // TODO: hook into Brian's seed() system for reproducibility
    static std::mt19937 _brian_cppyy_rng;

    // ── Helper to extract a C++ pointer from a PyCapsule ──
    // This is how we bridge Cython's DynamicArray objects to cppyy:
    // Cython wraps the C++ pointer in a PyCapsule, Python passes the
    // capsule to our function, and we unwrap it back to a C++ pointer.
    #include <Python.h>

    template<typename T>
    DynamicArray1D<T>* _extract_dynamic_array_1d(PyObject* capsule) {{
        void* ptr = PyCapsule_GetPointer(capsule, "DynamicArray1D");
        return static_cast<DynamicArray1D<T>*>(ptr);
    }}

    template<typename T>
    DynamicArray2D<T>* _extract_dynamic_array_2d(PyObject* capsule) {{
        void* ptr = PyCapsule_GetPointer(capsule, "DynamicArray2D");
        return static_cast<DynamicArray2D<T>*>(ptr);
    }}

    #endif // _BRIAN2_CPPYY_SUPPORT_CODE
    """
    cppyy.cppdef(guarded_code)
    _support_code_initialized = True


def _make_func_name(codeobj_name: str, block: str) -> str:
    """
    Build a deterministic C++ function name from code object + block name.
    Must match the Jinja2 template logic in common_group.cpp.
    """
    safe: str = codeobj_name.replace(".", "_").replace("*", "").replace("-", "_")
    return f"_brian_cppyy_{block}_{safe}"


def _cppyy_constant_or_scalar(varname: str, variable: Variable) -> str:
    """
    Like constant_or_scalar but uses _ptr_array_X naming to match our C++ params.

    The standard version produces "_array_X[0]" (device naming), but our
    function signatures use "_ptr_array_X" (generator naming).
    """
    if variable.array:
        return f"{CppyyCodeGenerator.get_array_name(variable)}[0]"
    else:
        return f"{varname}"


class CppyyCodeObject(NumpyCodeObject):
    """
    Code object that JIT-compiles C++ via cppyy/Cling.

    Inherits NumpyCodeObject's lifecycle but overrides namespace population
    to set up _ptr_array_* and _num* entries that our C++ functions expect.
    """

    templater: Templater = Templater(
        "brian2.codegen.runtime.cppyy_rt",
        ".cpp",
        env_globals={
            "c_data_type": _cppyy_c_data_type,
            "constant_or_scalar": _cppyy_constant_or_scalar,
        },
    )
    generator_class: type = CppyyCodeGenerator
    class_name: str = "cppyy"

    def __init__(
        self,
        owner: Any,
        code: Any,
        variables: dict[str, Variable],
        variable_indices: dict[str, str],
        template_name: str,
        template_source: str,
        compiler_kwds: dict[str, Any],
        name: str = "cppyy_code_object*",
    ) -> None:
        check_compiler_kwds(compiler_kwds, [], "cppyy")
        super().__init__(
            owner,
            code,
            variables,
            variable_indices,
            template_name,
            template_source,
            compiler_kwds={},
            name=name,
        )
        # Populated in compile() — maps block → parameter metadata
        self._param_mappings: dict[str, list[ParamTuple]] = {}
        # Prevent GC of arrays whose pointers are held by C++ globals
        self._namespace_refs: dict[str, NDArray[Any]] = {}

    @classmethod
    def is_available(cls) -> bool:
        """Check if cppyy is installed without importing it."""
        return importlib.util.find_spec("cppyy") is not None

    # --- Namespace population ---
    #
    # We override entirely (not calling super) because NumpyCodeObject
    # doesn't set _num* entries and uses device naming instead of generator naming.

    def variables_to_namespace(self) -> None:
        """
        Fill self.namespace with everything the C++ functions need.

        Arrays go under generator naming (_ptr_array_*), sizes under _num*,
        constants under their plain name, and Variable objects under _var_*.
        """
        self.nonconstant_values: list[NonconstantEntry] = []

        # Ensure _owner is available (needed for monitors in fallback path)
        if "_owner" not in self.namespace:
            self.namespace["_owner"] = self.owner

        for name, var in self.variables.items():
            if isinstance(var, Function):
                self._insert_func_namespace(var)
                continue

            if isinstance(var, (AuxiliaryVariable, Subexpression)):
                continue

            # Try to get the value — some dummy Variables don't have one
            try:
                if not hasattr(var, "get_value"):
                    raise TypeError()
                value: Any = var.get_value()
            except (TypeError, AttributeError):
                self.namespace[name] = var
                continue

            if isinstance(var, ArrayVariable):
                gen_name: str = self.generator_class.get_array_name(var)
                self.namespace[gen_name] = value
                self.namespace[f"_num{name}"] = var.get_len()

                # Scalar constants also get a plain-name entry with the unwrapped value
                if var.scalar and var.constant:
                    self.namespace[name] = value.item()
            else:
                self.namespace[name] = value

            # ── Dynamic arrays: store BOTH the data view AND the capsule ──
            # The data view (_ptr_array_*) gives C++ direct pointer access
            # to the current data buffer, used in computation functions.
            # The capsule (_capsule_*) gives C++ access to the DynamicArray
            # C++ object itself, used in monitor functions that need resize.
            if isinstance(var, DynamicArrayVariable):
                dyn_array_name = self.generator_class.get_array_name(
                    var, access_data=False
                )
                self.namespace[dyn_array_name] = self.device.get_value(
                    var, access_data=False
                )

                capsule_name = f"{dyn_array_name}_capsule"
                try:
                    capsule = self.device.get_capsule(var)
                    self.namespace[capsule_name] = capsule
                except (TypeError, AttributeError):
                    # Not all variables support capsules (e.g. plain arrays)
                    pass

            self.namespace[f"_var_{name}"] = var

            if isinstance(var, DynamicArrayVariable) and var.needs_reference_update:
                gen_name = self.generator_class.get_array_name(var)
                self.nonconstant_values.append((gen_name, var.get_value))
                self.nonconstant_values.append((f"_num{name}", var.get_len))

    def update_namespace(self) -> None:
        """Refresh data pointers/sizes for dynamic arrays that may have been resized."""
        for name, func in self.nonconstant_values:
            self.namespace[name] = func()

    def _insert_func_namespace(self, func: Function) -> None:
        """
        Pull in a function implementation's namespace (e.g. TimedArray data).
        Most built-in functions have nothing to inject; this is a no-op for them.
        """
        try:
            impl = func.implementations[self.__class__]
        except KeyError:
            return

        func_namespace: dict[str, Any] | None = impl.get_namespace(self.owner)
        if func_namespace is not None:
            self.namespace.update(func_namespace)

        if impl.dependencies is not None:
            for dep in impl.dependencies.values():
                self._insert_func_namespace(dep)

    # --- Parameter mapping ---
    #
    # Reconstructs the same param list the generator built in determine_keywords().
    # Both iterate sorted(self.variables.items()) with the same filtering, so order matches.

    def _build_param_mapping(self) -> list[ParamTuple]:
        """
        Build the (cpp_param_name, namespace_key, c_type) list matching the
        C++ function signature order.

        This MUST mirror the iteration logic in CppyyCodeGenerator.determine_keywords()
        exactly — same sorted order, same filtering, same parameter additions —
        otherwise the call-site args won't line up with the compiled signature.
        """
        params: list[ParamTuple] = []
        handled_pointers: set[str] = set()

        for varname, var in sorted(self.variables.items()):
            if isinstance(var, (AuxiliaryVariable, Subexpression)):
                continue
            if isinstance(var, Function):
                continue

            if isinstance(var, Constant):
                c_type: str = _cppyy_c_data_type(type(var.value))
                params.append((varname, varname, c_type))
                continue

            if isinstance(var, ArrayVariable):
                pointer_name: str = self.generator_class.get_array_name(var)
                if pointer_name in handled_pointers:
                    continue
                handled_pointers.add(pointer_name)

                if getattr(var, "ndim", 1) > 1:
                    # 2D dynamic arrays: pass capsule only (no data pointer).
                    # Mirrors determine_keywords() which does the same.
                    if isinstance(var, DynamicArrayVariable):
                        dyn_name = self.generator_class.get_array_name(
                            var, access_data=False
                        )
                        capsule_key = f"{dyn_name}_capsule"
                        params.append((capsule_key, capsule_key, "PyObject*"))
                    continue

                c_type = _cppyy_c_data_type(var.dtype)
                namespace_key: str = self.generator_class.get_array_name(var)

                params.append((pointer_name, namespace_key, f"{c_type}*"))

                if not var.scalar:
                    params.append((f"_num{varname}", f"_num{varname}", "int"))

                # 1D dynamic arrays: ALSO pass the capsule so C++ can resize.
                # This mirrors determine_keywords() which appends the capsule
                # param right after the pointer + size params.
                if isinstance(var, DynamicArrayVariable):
                    dyn_name = self.generator_class.get_array_name(
                        var, access_data=False
                    )
                    capsule_key = f"{dyn_name}_capsule"
                    params.append((capsule_key, capsule_key, "PyObject*"))

        return params

    # --- Compilation ---

    def compile_block(self, block: str) -> Any | None:
        """
        JIT-compile a code block and wire up any user-function globals.
        Returns the compiled function, or None for empty blocks.
        """
        code: str = getattr(self.code, block, "").strip()
        if not code or "EMPTY_CODE_BLOCK" in code:
            return None

        cppyy = _get_cppyy()
        _ensure_support_code()

        logger.diagnostic(f"cppyy: compiling '{block}' for {self.name}")
        try:
            cppyy.cppdef(code)
        except Exception as exc:
            raise BrianObjectException(
                f"cppyy compilation failed for '{block}' of '{self.name}'.\n"
                f"Generated C++ code:\n{code}\n",
                self.owner,
            ) from exc

        func_name: str = _make_func_name(self.name, block)
        try:
            compiled_func: Any = getattr(cppyy.gbl, func_name)
        except AttributeError:
            raise RuntimeError(
                f"cppyy compiled OK but function '{func_name}' not found. "
                f"Template/name mismatch? codeobj={self.name}, block={block}"
            ) from None

        # Wire up static C++ globals for user functions (e.g. TimedArray data pointers)
        self._set_user_func_globals(cppyy)

        self._param_mappings[block] = self._build_param_mapping()

        # register with introspector if enabled
        self._register_with_introspector(block, code)

        return compiled_func

    def _set_user_func_globals(self, cppyy: Any) -> None:
        """
        Point C++ static globals (e.g. `static double* _namespace_timedarray_values`)
        at the actual numpy data. Also pins the arrays to prevent GC.
        """
        for _name, var in self.variables.items():
            if not isinstance(var, Function):
                continue
            try:
                impl = var.implementations[self.__class__]
            except KeyError:
                continue

            func_namespace: dict[str, Any] | None = impl.get_namespace(self.owner)
            if not func_namespace:
                continue

            for ns_key, ns_value in func_namespace.items():
                if hasattr(ns_value, "dtype") and ns_value.ndim >= 1:
                    cpp_global_name: str = f"_namespace{ns_key}"
                    try:
                        setattr(cppyy.gbl, cpp_global_name, ns_value)
                        self._namespace_refs[ns_key] = ns_value
                        logger.diagnostic(
                            f"cppyy: set global {cpp_global_name} → "
                            f"array shape {ns_value.shape}"
                        )
                    except AttributeError:
                        logger.warn(
                            f"Could not set C++ global '{cpp_global_name}' for "
                            f"'{ns_key}'. May segfault if the function is called."
                        )

    def _register_with_introspector(self, block: str, source: str) -> None:
        """Register this code object with the global introspector, if enabled."""
        from .introspector import CppyyIntrospector

        introspector: CppyyIntrospector | None = CppyyIntrospector.get_instance()
        if introspector is not None:
            introspector.register(self, block, source)

    # --- Execution ---

    def run_block(self, block: str) -> None:
        """
        Call a compiled C++ function with args extracted from self.namespace.

        cppyy does the numpy→pointer conversion automatically: a float64 array
        passed where C++ expects double* gets its buffer pointer extracted with
        zero copies.
        """
        compiled_func: Any | None = self.compiled_code.get(block)
        if compiled_func is None:
            return

        try:
            param_mapping: list[ParamTuple] = self._param_mappings[block]
            args: list[Any] = []

            for cpp_name, ns_key, c_type in param_mapping:
                val: Any = self.namespace.get(ns_key)

                if val is None:
                    # Naming bridge bug — log and limp along with a zero
                    logger.warn(
                        f"Namespace key '{ns_key}' missing for param "
                        f"'{cpp_name}' ({c_type}) in {self.name}.{block}. "
                        f"Keys: {sorted(self.namespace.keys())[:20]}..."
                    )
                    if "*" in c_type:
                        args.append(np.zeros(1, dtype=np.float64))
                    else:
                        args.append(0)
                else:
                    if isinstance(val, np.ndarray):
                        val = np.ascontiguousarray(val)
                        # bool arrays need int8 view so cppyy's buffer protocol matches
                        if val.dtype == np.bool_:
                            val = val.view(np.int8)
                        # cppyy can't extract a buffer pointer from empty arrays —
                        # pass a 1-element dummy instead. The C++ code won't read
                        # past _num* elements anyway, and for dynamic arrays the
                        # real access goes through the capsule/DynamicArray object.
                        if val.size == 0 and c_type.endswith("*"):
                            val = np.zeros(1, dtype=val.dtype)
                    args.append(val)
            compiled_func(*args)

        except Exception as exc:
            raise BrianObjectException(
                f"Exception during '{block}' of '{self.name}'.\n",
                self.owner,
            ) from exc


codegen_targets.add(CppyyCodeObject)

# NOTE: rand/randn/clip/sign/timestep/poisson implementations are registered
# on CppyyCodeGenerator (in cppyy_generator.py), not here. This is intentional —
# the generator needs them during code generation, and FunctionImplementationContainer
# finds them via MRO fallback. Registering on both causes shadowing bugs.
