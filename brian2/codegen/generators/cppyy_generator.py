"""
C++ code generator for the cppyy runtime target.

Inherits CPPCodeGenerator's full translation pipeline (expressions, the
read→declare→execute→write phases, scalar hoisting, boolean optimization).
Overrides array naming and keyword generation so data arrives from Python
as function parameters rather than global C++ variables.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from brian2.codegen.generators.cpp_generator import (
    CPPCodeGenerator,
    c_data_type,
    stripped_deindented_lines,
)
from brian2.core.functions import DEFAULT_FUNCTIONS, Function
from brian2.core.variables import (
    ArrayVariable,
    AuxiliaryVariable,
    Constant,
    DynamicArrayVariable,
    Subexpression,
)

# (c_type, param_name, namespace_key)
FunctionParam = tuple[str, str, str]


def _extract_primary_cpp_symbol(piece: str) -> str | None:
    """
    Extract the primary C++ symbol name (function or variable) defined in a code piece.

    Only the FIRST non-comment, non-empty line is inspected so that identifiers
    inside function bodies are never mistaken for the declaration symbol.

    Returns None when no identifiable symbol is found.

    Examples::

        "static double* _namespaceta_values;"  -> "_namespaceta_values"
        "static inline double ta(..."          -> "ta"
        "static inline double _timedarray(...  -> "_timedarray"
    """
    for line in piece.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("/*"):
            continue
        # Collect all identifiers that appear immediately before '(', '[', or ';'
        # on this first declaration line, then take the last one — it is the
        # function/variable name, not the return-type keywords.
        candidates = re.findall(r"\b(\w+)\s*(?=[(\[;])", stripped)
        if candidates:
            return candidates[-1]
        return None
    return None


def _cppyy_c_data_type(dtype: type | Any) -> str:
    """
    Like c_data_type but remaps types for cppyy's buffer protocol.

    cppyy is strict about buffer types:
    - numpy int8 maps to signed char (int8_t), not char
    - numpy int64 maps to ``long`` on LP64 platforms (macOS, Linux 64-bit),
      but ``int64_t`` is typedef'd to ``long long`` — cppyy's buffer protocol
      only matches the canonical type, so we must use ``long`` directly.
    """
    import struct

    ctype: str = c_data_type(dtype)
    if ctype == "char":
        return "int8_t"
    # On LP64 platforms (sizeof(long)==8), cppyy maps numpy int64 buffers to
    # ``long*`` but rejects ``int64_t*`` (== ``long long*``).
    if ctype == "int64_t" and struct.calcsize("l") == 8:
        return "long"
    if ctype == "uint64_t" and struct.calcsize("l") == 8:
        return "unsigned long"
    return ctype


class CppyyCodeGenerator(CPPCodeGenerator):
    """
    C++ code generator targeting cppyy's JIT runtime.

    All C++ translation logic (expressions, 4-phase pattern, etc.) is inherited.
    We only change how arrays are named and how keywords/params are assembled.
    """

    class_name: str = "cppyy"

    @staticmethod
    def get_array_name(var: ArrayVariable, access_data: bool = True) -> str:
        """
        Globally unique name for an array variable.

        access_data=True  → "_ptr_array_{owner}_{name}"  (data pointer)
        access_data=False → "_dynamic_array_{owner}_{name}"  (container object)
        """
        owner_name: str = getattr(var.owner, "name", "temporary")

        if isinstance(var, DynamicArrayVariable):
            if access_data:
                return f"_ptr_array_{owner_name}_{var.name}"
            else:
                return f"_dynamic_array_{owner_name}_{var.name}"
        elif isinstance(var, ArrayVariable):
            return f"_ptr_array_{owner_name}_{var.name}"
        else:
            raise TypeError(
                f"get_array_name called with non-array variable: {type(var)}"
            )

    def determine_keywords(self) -> dict[str, Any]:
        """
        Build template keywords: function params, support code, hash defines.

        This runs at the end of translate_statement_sequence(). The returned
        dict gets merged with scalar_code/vector_code and passed to templates.

        We iterate sorted(self.variables.items()) — the code object's
        _build_param_mapping does the same, so parameter order is guaranteed
        to match between the signature and the call site.
        """

        support_code_parts: list[str] = []
        hash_define_parts: list[str] = []
        user_functions: list[Any] = []
        user_func_namespaces: dict[
            str, Any
        ] = {}  # for setting C++ globals post-compile
        added: set[str] = set()

        function_params: list[FunctionParam] = []
        handled_pointers: set[str] = set()

        for varname, var in sorted(self.variables.items()):
            if isinstance(var, (AuxiliaryVariable, Subexpression)):
                continue

            # --- User functions (TimedArray, BinomialFunction, etc.) ---
            if isinstance(var, Function):
                if self.codeobj_class in var.implementations:
                    result: tuple | None = self._add_user_function(varname, var, added)
                    if result is not None:
                        hd, _pointers, sc, uf = result
                        hash_define_parts.extend(hd)
                        # Wrap each user-function support code piece in its own
                        # #ifndef guard so that Cling doesn't redeclare static
                        # symbols (e.g. _namespace_timedarray_values) when the
                        # same function is used across multiple code objects.
                        for piece in sc:
                            stripped = "\n".join(
                                line
                                for line in piece.split("\n")
                                if line.strip() and not line.strip().startswith("//")
                            )
                            if stripped:
                                # Key the guard on the C++ symbol name so that
                                # Cling never tries to redefine the same symbol
                                # even when the body differs (e.g. a GC'd
                                # TimedArray's name is reused with different K/N
                                # in a later test).  The data pointer is always
                                # refreshed by _set_user_func_globals after
                                # compilation, so skipping the redeclaration is
                                # safe as long as only one test runs at a time.
                                symbol = _extract_primary_cpp_symbol(piece)
                                if symbol:
                                    guard = f"_BRIAN_CPPYY_SYM_{symbol}"
                                else:
                                    h = hashlib.md5(stripped.encode()).hexdigest()[:16]
                                    guard = f"_BRIAN_CPPYY_UF_{h}"
                                support_code_parts.append(
                                    f"#ifndef {guard}\n#define {guard}\n{piece}\n"
                                    f"#endif // {guard}"
                                )
                            else:
                                support_code_parts.append(piece)
                        user_functions.extend(uf)

                    # Grab namespace values (actual numpy arrays) for C++ globals
                    impl = var.implementations[self.codeobj_class]
                    func_ns: dict[str, Any] | None = impl.get_namespace(self.owner)
                    if func_ns:
                        user_func_namespaces.update(func_ns)
                continue

            # --- Constants: scalar typed parameters ---
            if isinstance(var, Constant):
                c_type: str = _cppyy_c_data_type(type(var.value))
                function_params.append((c_type, varname, varname))
                continue

            # --- Array variables: pointer + size parameters ---
            if isinstance(var, ArrayVariable):
                pointer_name = self.get_array_name(var)
                if pointer_name in handled_pointers:
                    continue
                handled_pointers.add(pointer_name)

                if getattr(var, "ndim", 1) > 1:
                    # 2D dynamic arrays: pass the capsule instead of a data pointer,
                    # because monitors need to resize them. The C++ code extracts
                    # the DynamicArray2D<T>* from the capsule and calls methods on it.
                    if isinstance(var, DynamicArrayVariable):
                        dyn_name = self.get_array_name(var, access_data=False)
                        capsule_key = f"{dyn_name}_capsule"
                        function_params.append(("PyObject*", capsule_key, capsule_key))
                    continue

                c_type = _cppyy_c_data_type(var.dtype)
                namespace_key = self.get_array_name(var)
                function_params.append((f"{c_type}*", pointer_name, namespace_key))

                if not var.scalar:
                    function_params.append(("int", f"_num{varname}", f"_num{varname}"))

                # For 1D dynamic arrays, ALSO pass the capsule so monitors can resize
                if isinstance(var, DynamicArrayVariable):
                    dyn_name = self.get_array_name(var, access_data=False)
                    capsule_key = f"{dyn_name}_capsule"
                    function_params.append(("PyObject*", capsule_key, capsule_key))

        # --- Object variables with capsule-like names (e.g. _queue_capsule) ---
        # These are PyCapsule objects passed as PyObject* parameters.
        for varname, var in sorted(self.variables.items()):
            if varname.endswith("_capsule") and not isinstance(
                var,
                (ArrayVariable, Constant, Function, AuxiliaryVariable, Subexpression),
            ):
                if varname not in {p[1] for p in function_params}:
                    function_params.append(("PyObject*", varname, varname))

        # group_get_indices: both _cond and _indices are AuxiliaryVariables only
        # when the IndexWrapper.__getitem__ path in group.py creates the code
        # object.  Other templates (e.g. synapses_create_generator) also have
        # _cond but not _indices.  Require both to uniquely identify this template.
        if isinstance(self.variables.get("_cond"), AuxiliaryVariable) and isinstance(
            self.variables.get("_indices"), AuxiliaryVariable
        ):
            function_params.append(("int*", "_return_values_buf", "_return_values_buf"))
            function_params.append(("int*", "_return_values_n", "_return_values_n"))

        # group_variable_get: _variable AuxiliaryVariable + _group_idx array present.
        # C++ writes subexpression values per index into _output_buf.
        if isinstance(self.variables.get("_variable"), AuxiliaryVariable) and (
            "_group_idx" in self.variables
            and not isinstance(self.variables.get("_cond"), AuxiliaryVariable)
        ):
            var = self.variables["_variable"]
            c_type = _cppyy_c_data_type(var.dtype)
            function_params.append((f"{c_type}*", "_output_buf", "_output_buf"))

        # group_variable_get_conditional: _variable and _cond are AuxiliaryVariables,
        # but _indices is NOT (unlike group_get_indices).
        # C++ writes matching values into _output_buf and the count into _output_n[0].
        if (
            isinstance(self.variables.get("_variable"), AuxiliaryVariable)
            and isinstance(self.variables.get("_cond"), AuxiliaryVariable)
            and not isinstance(self.variables.get("_indices"), AuxiliaryVariable)
        ):
            var = self.variables["_variable"]
            c_type = _cppyy_c_data_type(var.dtype)
            function_params.append((f"{c_type}*", "_output_buf", "_output_buf"))
            function_params.append(("int*", "_output_n", "_output_n"))

        # Optional denormals flushing (gcc/clang x86)
        denormals_code: str = ""
        if self.flush_denormals:
            denormals_code = """
            #define CSR_FLUSH_TO_ZERO (1 << 15)
            unsigned csr = __builtin_ia32_stmxcsr();
            csr |= CSR_FLUSH_TO_ZERO;
            __builtin_ia32_ldmxcsr(csr);
            """

        return {
            "support_code_lines": "\n".join(
                stripped_deindented_lines("\n".join(support_code_parts))
            ),
            "hashdefine_lines": "\n".join(
                stripped_deindented_lines("\n".join(hash_define_parts))
            ),
            "denormals_code_lines": "\n".join(
                stripped_deindented_lines(denormals_code)
            ),
            "function_params": function_params,
            "user_func_namespaces": user_func_namespaces,
            "user_functions": user_functions,
        }


# --- Function implementations ---
#
# We get sin/cos/exp/log/etc. for free via MRO (registered on CPPCodeGenerator).
# Same for arcsin→asin, int→int_, exprel, TimedArray, BinomialFunction.
#
# clip/sign/timestep/poisson/rand/randn are defined globally in _ensure_support_code()
# (cppyy_rt.py) so each code object emits no per-codeobject support code for them.
# This prevents redefinition errors when different code objects share the same
# function but differ in other support code (different hash → both would compile the
# function body without this guard).

DEFAULT_FUNCTIONS["clip"].implementations.add_implementation(
    CppyyCodeGenerator, code="", name="_clip"
)
DEFAULT_FUNCTIONS["sign"].implementations.add_implementation(
    CppyyCodeGenerator, code="", name="_sign"
)
DEFAULT_FUNCTIONS["timestep"].implementations.add_implementation(
    CppyyCodeGenerator, code="", name="_timestep"
)
DEFAULT_FUNCTIONS["poisson"].implementations.add_implementation(
    CppyyCodeGenerator, code="", name="_poisson"
)
DEFAULT_FUNCTIONS["rand"].implementations.add_dynamic_implementation(
    CppyyCodeGenerator,
    code=lambda owner: {},
    namespace=lambda owner: {},
    name="_rand",
)
DEFAULT_FUNCTIONS["randn"].implementations.add_dynamic_implementation(
    CppyyCodeGenerator,
    code=lambda owner: {},
    namespace=lambda owner: {},
    name="_randn",
)
