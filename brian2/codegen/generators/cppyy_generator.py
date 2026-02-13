"""
C++ code generator for the cppyy runtime target.

Inherits CPPCodeGenerator's full translation pipeline (expressions, the
read→declare→execute→write phases, scalar hoisting, boolean optimization).
Overrides array naming and keyword generation so data arrives from Python
as function parameters rather than global C++ variables.
"""

from __future__ import annotations

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


def _cppyy_c_data_type(dtype: type | Any) -> str:
    """
    Like c_data_type but maps bool→int8_t instead of char.

    cppyy is strict about buffer types: numpy int8 maps to signed char (int8_t),
    not char. Using int8_t in the signature lets the buffer protocol match.
    The function body still uses char for locals — implicit conversion handles it.
    """
    ctype: str = c_data_type(dtype)
    if ctype == "char":
        return "int8_t"
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
        from brian2.devices.device import get_device

        device: Any = get_device()

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
                        support_code_parts.extend(sc)
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
                pointer_name: str = self.get_array_name(var)
                if pointer_name in handled_pointers:
                    continue
                handled_pointers.add(pointer_name)

                # Skip multidimensional dynamic arrays (need special handling)
                if getattr(var, "ndim", 1) > 1:
                    continue

                c_type = _cppyy_c_data_type(var.dtype)
                namespace_key: str = device.get_array_name(var)

                function_params.append((f"{c_type}*", pointer_name, namespace_key))

                if not var.scalar:
                    function_params.append(("int", f"_num{varname}", f"_num{varname}"))

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
# We must explicitly register clip/sign/timestep/poisson — they're only on
# CythonCodeGenerator which isn't in our MRO chain.

_clip_code: str = """
template<typename T>
inline T _clip(T value, double a_min, double a_max) {
    if (value < (T)a_min) return (T)a_min;
    if (value > (T)a_max) return (T)a_max;
    return value;
}
"""
DEFAULT_FUNCTIONS["clip"].implementations.add_implementation(
    CppyyCodeGenerator, code=_clip_code, name="_clip"
)

_sign_code: str = """
template<typename T>
inline int _sign(T x) {
    return (T(0) < x) - (x < T(0));
}
"""
DEFAULT_FUNCTIONS["sign"].implementations.add_implementation(
    CppyyCodeGenerator, code=_sign_code, name="_sign"
)

_timestep_code: str = """
inline int64_t _timestep(double t, double dt) {
    return (int64_t)((t + 1e-3*dt)/dt);
}
"""
DEFAULT_FUNCTIONS["timestep"].implementations.add_implementation(
    CppyyCodeGenerator, code=_timestep_code, name="_timestep"
)

_poisson_code: str = """
#include <random>
inline int32_t _poisson(double lam, int _vectorisation_idx) {
    std::poisson_distribution<int32_t> _poisson_dist(lam);
    return _poisson_dist(_brian_cppyy_rng);
}
"""
DEFAULT_FUNCTIONS["poisson"].implementations.add_implementation(
    CppyyCodeGenerator, code=_poisson_code, name="_poisson"
)

# rand/randn use the shared MT19937 engine from _ensure_support_code()
_rand_support: str = """
inline double _rand(const int _vectorisation_idx) {
    static std::uniform_real_distribution<double> _dist_rand(0.0, 1.0);
    return _dist_rand(_brian_cppyy_rng);
}
"""

_randn_support: str = """
inline double _randn(const int _vectorisation_idx) {
    static std::normal_distribution<double> _dist_randn(0.0, 1.0);
    return _dist_randn(_brian_cppyy_rng);
}
"""

DEFAULT_FUNCTIONS["rand"].implementations.add_dynamic_implementation(
    CppyyCodeGenerator,
    code=lambda owner: {"support_code": _rand_support},
    namespace=lambda owner: {},
    name="_rand",
)

DEFAULT_FUNCTIONS["randn"].implementations.add_dynamic_implementation(
    CppyyCodeGenerator,
    code=lambda owner: {"support_code": _randn_support},
    namespace=lambda owner: {},
    name="_randn",
)
