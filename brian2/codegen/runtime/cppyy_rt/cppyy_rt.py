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

import hashlib
import importlib.util
import os
import re
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
from ...generators.cppyy_generator import CppyyCodeGenerator, _cppyy_c_data_type
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


def _guard_support_code(code: str) -> str:
    """
    Wrap per-codeobject support code in #ifndef guards to prevent
    Cling redefinition errors.

    When Brian2 calls run() multiple times, it recreates code objects that
    generate identical support code (inline functions like _timestep, _rand).
    Cling can't redefine symbols, but it dores preserve preprocessor state
    across cppyy.cppdef() calls. So we wrap the support code in #ifndef
    guards keyed by a content hash — if Cling already compiled this exact
    block, the preprocessor skips it.

    The generated code has a predictable structure:
        // Per-codeobject support code
        [hash defines]
        [inline function definitions]    # this part gets guarded
        // Template-specific support code
        extern "C" void _brian_cppyy_...(...) { ... }

    We split at 'extern "C"', guard everything before it, and leave the
    function definition (which has a unique name) unguarded.
    """
    marker: str = 'extern "C"'
    pos: int = code.find(marker)
    if pos == -1:
        # No function definition found — nothing to guard
        return code

    support: str = code[:pos]
    func_def: str = code[pos:]

    # Check if there's actual compilable code (not just comments/whitespace)
    real_lines: list[str] = [
        line.strip()
        for line in support.split("\n")
        if line.strip() and not line.strip().startswith("//")
    ]
    if not real_lines:
        # Only comments before extern "C" — no risk of redefinition
        return code

    content_hash: str = hashlib.md5("\n".join(real_lines).encode()).hexdigest()[:16]
    guard: str = f"_BRIAN_CPPYY_SC_{content_hash}"

    return f"#ifndef {guard}\n#define {guard}\n{support}#endif // {guard}\n\n{func_def}"


# Maps C++ user-function name -> content_hash of the first compiled body.
# Used by _rename_conflicting_user_functions to detect when the same C++
# function name would be redefined with a different body in Cling.
_user_func_registry: dict[str, str] = {}

# Regex matching #ifndef _BRIAN_CPPYY_SYM_XXX ... #endif blocks emitted
# by the generator for each user-function support code piece.
_SYM_BLOCK_RE = re.compile(
    r"(#ifndef (_BRIAN_CPPYY_SYM_(\w+))\n"
    r"#define _BRIAN_CPPYY_SYM_\3\n"
    r"(.*?)"
    r"#endif // _BRIAN_CPPYY_SYM_\3)",
    re.DOTALL,
)


def _rename_conflicting_user_functions(code: str) -> tuple[str, dict[str, str]]:
    """
    Ensure that user-function definitions (TimedArray, BinomialFunction, …)
    in the per-codeobject support code never conflict with already-compiled
    Cling symbols.

    Strategy:
    - Scan for ``#ifndef _BRIAN_CPPYY_SYM_*`` guard blocks.
    - For blocks that contain a C++ function definition (detected by ``{``),
      compute a hash of the function body.
    - If the C++ function name was previously compiled with a *different* body,
      rename it to ``funcname_<hash>`` everywhere in this code string (both the
      definition and every call site inside the ``extern "C"`` body).
    - Also rename any associated ``_namespace_<funcname>_values`` global so that
      cppyy does not reject re-assigning a different-sized buffer to the same
      C++ pointer (cppyy tracks buffer sizes per global).
    - Update all guard macros accordingly.

    Returns the modified code and a dict mapping original C++ global names to
    renamed ones (used by ``_set_user_func_globals`` to write to the right symbol).

    This handles the case where Brian2 recycles a name like ``_timedarray`` after
    GC (the previous TimedArray with 2 values is collected, then a new one with
    10 values reuses the name).  Without renaming both would try to define the
    same C++ symbol, causing a Cling ``redefinition`` error.
    """
    global _user_func_registry

    renames: dict[str, str] = {}  # original_func_name -> new_func_name

    for m in _SYM_BLOCK_RE.finditer(code):
        cpp_symbol: str = m.group(3)  # e.g. "_timedarray"
        block_content: str = m.group(4).strip()

        # Only care about function definitions (have { in the content)
        if "{" not in block_content:
            continue

        content_hash: str = hashlib.md5(block_content.encode()).hexdigest()[:8]

        if cpp_symbol not in _user_func_registry:
            _user_func_registry[cpp_symbol] = content_hash
        elif _user_func_registry[cpp_symbol] != content_hash:
            # Same C++ name, different body → rename to avoid redefinition
            new_name = f"{cpp_symbol}_{content_hash}"
            if new_name not in _user_func_registry:
                _user_func_registry[new_name] = content_hash
            renames[cpp_symbol] = new_name

    if not renames:
        return code, {}

    # Maps original C++ global name → renamed global name.
    # Returned so that _set_user_func_globals can target the correct symbol.
    ns_global_renames: dict[str, str] = {}

    for old_name, new_name in renames.items():
        # 1. Rename call sites and function definitions.
        #    Negative lookbehind (?<![_\w]) skips occurrences that are already
        #    part of a longer identifier (e.g. _namespace_timedarray_values).
        code = re.sub(
            rf"(?<![_\w]){re.escape(old_name)}\b",
            new_name,
            code,
        )
        # Update guard macro suffix for the function (guard names are preceded by
        # "SYM_", which the lookbehind above skips).
        code = code.replace(f"SYM_{old_name}", f"SYM_{new_name}")

        # 2. Rename the associated _namespace_<funcname>_values global.
        #    cppyy tracks the buffer size of each C++ global; reassigning to a
        #    different-sized array raises "buffer too large for value".  Renaming
        #    the global gives each distinct function body its own C++ pointer so
        #    cppyy never sees a size mismatch.
        #
        #    Convention (from _add_user_function / generate_cpp_code):
        #       C++ global = "_namespace" + ns_key   where ns_key = "<funcname>_values"
        #    e.g. _timedarray → _namespace_timedarray_values
        old_ns_global = f"_namespace{old_name}_values"  # _namespace_timedarray_values
        new_ns_global = (
            f"_namespace{new_name}_values"  # _namespace_timedarray_H2_values
        )
        if old_ns_global in code:
            code = code.replace(old_ns_global, new_ns_global)
            ns_global_renames[old_ns_global] = new_ns_global

    return code, ns_global_renames


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


# --- One-time support code init ---
_support_code_initialized: bool = False

# --- Per-compilation unique counter (prevents Cling redefinition of extern "C" symbols) ---
_compile_counter: int = 0


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

    # Also add the synapses directory for spikequeue.h
    synapses_path = os.path.join(os.path.dirname(brian2.__file__), "synapses")
    cppyy.add_include_path(synapses_path)

    # Include the header — Cling compiles it and knows the class layout.
    # After this, cppyy C++ code can use DynamicArray1D<double>*, etc.
    cppyy.include("dynamic_array.h")
    cppyy.include("spikequeue.h")
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

    // int_() — stdint_compat.h may already define this (included by spikequeue.h)
    #ifndef _BRIAN_STDINT_COMPAT_H
    template<typename T>
    inline int32_t int_(T value) {{ return static_cast<int32_t>(value); }}
    #endif

    // Shared RNG for rand/randn/poisson
    static std::mt19937 _brian_cppyy_rng;

    // Seeding function callable from Python via cppyy.gbl._brian_cppyy_seed()
    extern "C" void _brian_cppyy_seed(unsigned int seed) {{
        _brian_cppyy_rng.seed(seed);
    }}
    extern "C" void _brian_cppyy_seed_random() {{
        std::random_device rd;
        _brian_cppyy_rng.seed(rd());
    }}

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

    // ── Helper to extract a CSpikeQueue from a PyCapsule ──
    inline CSpikeQueue* _extract_spike_queue(PyObject* capsule) {{
        void* ptr = PyCapsule_GetPointer(capsule, "CSpikeQueue");
        return static_cast<CSpikeQueue*>(ptr);
    }}

    // ── Global inline helpers (shared across all code objects) ──
    template<typename T>
    inline T _clip(T value, double a_min, double a_max) {{
        if (value < (T)a_min) return (T)a_min;
        if (value > (T)a_max) return (T)a_max;
        return value;
    }}

    template<typename T>
    inline int _sign(T x) {{
        return (T(0) < x) - (x < T(0));
    }}

    inline int64_t _timestep(double t, double dt) {{
        return (int64_t)((t + 1e-3*dt)/dt);
    }}

    inline int32_t _poisson(double lam, int _vectorisation_idx) {{
        std::poisson_distribution<int32_t> _poisson_dist(lam);
        return _poisson_dist(_brian_cppyy_rng);
    }}

    inline double _rand(const int _vectorisation_idx) {{
        static std::uniform_real_distribution<double> _dist_rand(0.0, 1.0);
        return _dist_rand(_brian_cppyy_rng);
    }}

    inline double _randn(const int _vectorisation_idx) {{
        static std::normal_distribution<double> _dist_randn(0.0, 1.0);
        return _dist_randn(_brian_cppyy_rng);
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
        # Maps block → unique C++ function name (counter-suffixed to avoid Cling redefinition)
        self._compiled_func_names: dict[str, str] = {}

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

        # group_get_indices: inject output buffers for the result.
        # C++ fills _return_values_buf and writes the match count to
        # _return_values_n[0].  run_block() reads them back as a return value.
        if self.template_name == "group_get_indices":
            N = int(self.namespace.get("N", 0))
            self.namespace["_return_values_buf"] = np.zeros(N, dtype=np.int32)
            self.namespace["_return_values_n"] = np.zeros(1, dtype=np.int32)

        # group_variable_get: C++ writes subexpression values per index into _output_buf.
        # Size = number of indices (_num_group_idx); dtype from _variable.
        if self.template_name == "group_variable_get":
            var = self.variables.get("_variable")
            n = int(self.namespace.get("_num_group_idx", 0))
            dtype = var.dtype if var is not None else np.float64
            self.namespace["_output_buf"] = np.zeros(max(n, 1), dtype=dtype)

        # group_variable_get_conditional: C++ writes matching values into _output_buf
        # and the count into _output_n[0].  Max size = N.
        if self.template_name == "group_variable_get_conditional":
            var = self.variables.get("_variable")
            N = int(self.namespace.get("N", 0))
            dtype = var.dtype if var is not None else np.float64
            self.namespace["_output_buf"] = np.zeros(max(N, 1), dtype=dtype)
            self.namespace["_output_n"] = np.zeros(1, dtype=np.int32)

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

        # --- Object variables with capsule-like names (e.g. _queue_capsule) ---
        handled_keys = {p[1] for p in params}
        for varname, var in sorted(self.variables.items()):
            if varname.endswith("_capsule") and not isinstance(
                var,
                (
                    ArrayVariable,
                    Constant,
                    Function,
                    AuxiliaryVariable,
                    Subexpression,
                ),
            ):
                if varname not in handled_keys:
                    params.append((varname, varname, "PyObject*"))

        # group_get_indices: append output-buffer params that mirror the extra
        # entries added by CppyyCodeGenerator.determine_keywords().
        if self.template_name == "group_get_indices":
            params.append(("_return_values_buf", "_return_values_buf", "int*"))
            params.append(("_return_values_n", "_return_values_n", "int*"))

        # group_variable_get: output buffer for subexpression values.
        if self.template_name == "group_variable_get":
            var = self.variables.get("_variable")
            dtype = var.dtype if var is not None else np.float64
            c_type = _cppyy_c_data_type(dtype)
            params.append(("_output_buf", "_output_buf", f"{c_type}*"))

        # group_variable_get_conditional: output buffer + count for conditional get.
        if self.template_name == "group_variable_get_conditional":
            var = self.variables.get("_variable")
            dtype = var.dtype if var is not None else np.float64
            c_type = _cppyy_c_data_type(dtype)
            params.append(("_output_buf", "_output_buf", f"{c_type}*"))
            params.append(("_output_n", "_output_n", "int*"))

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

        # Rename user functions whose name would be reused with a different body
        # (e.g. a GC'd TimedArray's C++ name is recycled by a new TimedArray
        # with different K/N parameters).  Must happen before _guard_support_code
        # so the outer hash is computed on the already-renamed code.
        code, _ns_renames = _rename_conflicting_user_functions(code)
        # Store so _set_user_func_globals knows which C++ globals were renamed.
        self._current_ns_global_renames: dict[str, str] = _ns_renames

        # Guard support code against redefinition (happens when run() is
        # called multiple times — Brian2 recreates code objects with the
        # same inline function definitions)
        code = _guard_support_code(code)

        # Make function name unique per-compilation to prevent Cling redefinition
        # errors when the same Brian object name appears across multiple test runs
        # or simulation setups in the same Python process.
        global _compile_counter
        original_func_name = _make_func_name(self.name, block)
        unique_func_name = f"{original_func_name}_{_compile_counter:06d}"
        _compile_counter += 1
        code = code.replace(original_func_name, unique_func_name)

        logger.diagnostic(f"cppyy: compiling '{block}' for {self.name}")
        try:
            cppyy.cppdef(code)
        except Exception as exc:
            raise BrianObjectException(
                f"cppyy compilation failed for '{block}' of '{self.name}'.\n"
                f"Generated C++ code:\n{code}\n",
                self.owner,
            ) from exc

        try:
            compiled_func: Any = getattr(cppyy.gbl, unique_func_name)
        except AttributeError:
            raise RuntimeError(
                f"cppyy compiled OK but function '{unique_func_name}' not found. "
                f"Template/name mismatch? codeobj={self.name}, block={block}"
            ) from None

        self._compiled_func_names[block] = unique_func_name

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
                    # If this global was renamed during compilation (because its
                    # function body differed from a previously compiled version),
                    # use the renamed symbol so we don't hit a size-mismatch error.
                    ns_renames = getattr(self, "_current_ns_global_renames", {})
                    cpp_global_name = ns_renames.get(cpp_global_name, cpp_global_name)
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

            # Sanity check: param count must match function arity
            expected_nargs = len(param_mapping)
            logger.diagnostic(
                f"cppyy: calling {self.name}.{block} with {expected_nargs} params"
            )

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
            try:
                compiled_func(*args)
            except Exception as cpp_exc:
                # Convert C++ out_of_range to Python IndexError so that
                # exc_isinstance(exc, IndexError) works in tests.
                cppyy_mod = _get_cppyy()
                if cppyy_mod is not None and isinstance(
                    cpp_exc, cppyy_mod.gbl.std.out_of_range
                ):
                    raise IndexError(str(cpp_exc)) from cpp_exc
                raise

            # group_get_indices: C++ wrote matching indices into the output
            # buffer and the count into _return_values_n[0].  Return the slice
            # so the caller (group.__getitem__) gets back a numpy int32 array.
            if self.template_name == "group_get_indices":
                n = int(self.namespace["_return_values_n"][0])
                return self.namespace["_return_values_buf"][:n].copy()

            # group_variable_get: C++ filled _output_buf with _num_group_idx values.
            if self.template_name == "group_variable_get":
                n = int(self.namespace.get("_num_group_idx", 0))
                return self.namespace["_output_buf"][:n].copy()

            # group_variable_get_conditional: C++ filled _output_buf with values
            # where _cond was True; count is in _output_n[0].
            if self.template_name == "group_variable_get_conditional":
                n = int(self.namespace["_output_n"][0])
                return self.namespace["_output_buf"][:n].copy()

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
