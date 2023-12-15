import itertools

import numpy

from brian2.codegen.cpp_prefs import C99Check
from brian2.core.functions import DEFAULT_FUNCTIONS, Function
from brian2.core.preferences import BrianPreference, prefs
from brian2.core.variables import ArrayVariable
from brian2.parsing.rendering import CPPNodeRenderer
from brian2.utils.logger import get_logger
from brian2.utils.stringtools import (
    deindent,
    stripped_deindented_lines,
    word_substitute,
)

from .base import CodeGenerator

logger = get_logger(__name__)

__all__ = ["CPPCodeGenerator", "c_data_type"]


def c_data_type(dtype):
    """
    Gives the C language specifier for numpy data types. For example,
    ``numpy.int32`` maps to ``int32_t`` in C.
    """
    # this handles the case where int is specified, it will be int32 or int64
    # depending on platform
    if dtype is int:
        dtype = numpy.array([1]).dtype.type
    if dtype is float:
        dtype = numpy.array([1.0]).dtype.type

    if dtype == numpy.float32:
        dtype = "float"
    elif dtype == numpy.float64:
        dtype = "double"
    elif dtype == numpy.int8:
        dtype = "int8_t"
    elif dtype == numpy.int16:
        dtype = "int16_t"
    elif dtype == numpy.int32:
        dtype = "int32_t"
    elif dtype == numpy.int64:
        dtype = "int64_t"
    elif dtype == numpy.uint16:
        dtype = "uint16_t"
    elif dtype == numpy.uint32:
        dtype = "uint32_t"
    elif dtype == numpy.uint64:
        dtype = "uint64_t"
    elif dtype == numpy.bool_ or dtype is bool:
        dtype = "char"
    else:
        raise ValueError(f"dtype {str(dtype)} not known.")
    return dtype


# Preferences
prefs.register_preferences(
    "codegen.generators.cpp",
    "C++ codegen preferences",
    restrict_keyword=BrianPreference(
        default="__restrict",
        docs="""
        The keyword used for the given compiler to declare pointers as restricted.

        This keyword is different on different compilers, the default works for
        gcc and MSVS.
        """,
    ),
    flush_denormals=BrianPreference(
        default=False,
        docs="""
        Adds code to flush denormals to zero.

        The code is gcc and architecture specific, so may not compile on all
        platforms. The code, for reference is::

            #define CSR_FLUSH_TO_ZERO         (1 << 15)
            unsigned csr = __builtin_ia32_stmxcsr();
            csr |= CSR_FLUSH_TO_ZERO;
            __builtin_ia32_ldmxcsr(csr);

        Found at `<http://stackoverflow.com/questions/2487653/avoiding-denormal-values-in-c>`_.
        """,
    ),
)


typestrs = ["int32_t", "int64_t", "float", "double", "long double"]
hightype_support_code = "template < typename T1, typename T2 > struct _higher_type;\n"
for ix, xtype in enumerate(typestrs):
    for iy, ytype in enumerate(typestrs):
        hightype = typestrs[max(ix, iy)]
        hightype_support_code += f"""
template < > struct _higher_type<{xtype},{ytype}> {{ typedef {hightype} type; }};
        """

mod_support_code = """
// General template, used for floating point types
template < typename T1, typename T2 >
static inline typename _higher_type<T1,T2>::type
_brian_mod(T1 x, T2 y)
{
    return x-y*floor(1.0*x/y);
}

// Specific implementations for integer types
// (from Cython, see LICENSE file)
template <>
inline int32_t _brian_mod(int32_t x, int32_t y)
{
    int32_t r = x % y;
    r += ((r != 0) & ((r ^ y) < 0)) * y;
    return r;
}

template <>
inline int64_t _brian_mod(int32_t x, int64_t y)
{
    int64_t r = x % y;
    r += ((r != 0) & ((r ^ y) < 0)) * y;
    return r;
}

template <>
inline int64_t _brian_mod(int64_t x, int32_t y)
{
    int64_t r = x % y;
    r += ((r != 0) & ((r ^ y) < 0)) * y;
    return r;
}

template <>
inline int64_t _brian_mod(int64_t x, int64_t y)
{
    int64_t r = x % y;
    r += ((r != 0) & ((r ^ y) < 0)) * y;
    return r;
}
"""

floordiv_support_code = """
// General implementation, used for floating point types
template < typename T1, typename T2 >
static inline typename _higher_type<T1,T2>::type
_brian_floordiv(T1 x, T2 y)
{{
    return floor(1.0*x/y);
}}

// Specific implementations for integer types
// (from Cython, see LICENSE file)
template <>
inline int32_t _brian_floordiv<int32_t, int32_t>(int32_t a, int32_t b) {
    int32_t q = a / b;
    int32_t r = a - q*b;
    q -= ((r != 0) & ((r ^ b) < 0));
    return q;
}
template <>
inline int64_t _brian_floordiv<int32_t, int64_t>(int32_t a, int64_t b) {
    int64_t q = a / b;
    int64_t r = a - q*b;
    q -= ((r != 0) & ((r ^ b) < 0));
    return q;
}
template <>
inline int64_t _brian_floordiv<int64_t, int>(int64_t a, int32_t b) {
    int64_t q = a / b;
    int64_t r = a - q*b;
    q -= ((r != 0) & ((r ^ b) < 0));
    return q;
}
template <>
inline int64_t _brian_floordiv<int64_t, int64_t>(int64_t a, int64_t b) {
    int64_t q = a / b;
    int64_t r = a - q*b;
    q -= ((r != 0) & ((r ^ b) < 0));
    return q;
}
"""

pow_support_code = """
#ifdef _MSC_VER
#define _brian_pow(x, y) (pow((double)(x), (y)))
#else
#define _brian_pow(x, y) (pow((x), (y)))
#endif
"""

_universal_support_code = (
    hightype_support_code + mod_support_code + floordiv_support_code + pow_support_code
)


class CPPCodeGenerator(CodeGenerator):
    """
    C++ language

    C++ code templates should provide Jinja2 macros with the following names:

    ``main``
        The main loop.
    ``support_code``
        The support code (function definitions, etc.), compiled in a separate
        file.

    For user-defined functions, there are two keys to provide:

    ``support_code``
        The function definition which will be added to the support code.
    ``hashdefine_code``
        The ``#define`` code added to the main loop.

    See `TimedArray` for an example of these keys.
    """

    class_name = "cpp"

    universal_support_code = _universal_support_code

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.c_data_type = c_data_type

    @property
    def restrict(self):
        return f"{prefs['codegen.generators.cpp.restrict_keyword']} "

    @property
    def flush_denormals(self):
        return prefs["codegen.generators.cpp.flush_denormals"]

    @staticmethod
    def get_array_name(var, access_data=True):
        # We have to do the import here to avoid circular import dependencies.
        from brian2.devices.device import get_device

        device = get_device()
        if access_data:
            return f"_ptr{device.get_array_name(var)}"
        else:
            return device.get_array_name(var, access_data=False)

    def translate_expression(self, expr):
        expr = word_substitute(expr, self.func_name_replacements)
        return (
            CPPNodeRenderer(auto_vectorise=self.auto_vectorise)
            .render_expr(expr)
            .strip()
        )

    def translate_statement(self, statement):
        var, op, expr, comment = (
            statement.var,
            statement.op,
            statement.expr,
            statement.comment,
        )
        # For C++ we replace complex expressions involving boolean variables into a sequence of
        # if/then expressions with simpler expressions. This is provided by the optimise_statements
        # function.
        if statement.used_boolean_variables is not None and len(
            statement.used_boolean_variables
        ):
            used_boolvars = statement.used_boolean_variables
            bool_simp = statement.boolean_simplified_expressions
            if op == ":=":
                # we have to declare the variable outside the if/then statement (which
                # unfortunately means we can't make it const but the optimisation is worth
                # it anyway).
                codelines = [f"{self.c_data_type(statement.dtype)} {var};"]
                op = "="
            else:
                codelines = []
            firstline = True
            # bool assigns is a sequence of (var, value) pairs giving the conditions under
            # which the simplified expression simp_expr holds
            for bool_assigns, simp_expr in bool_simp.items():
                # generate a boolean expression like ``var1 && var2 && !var3``
                atomics = []
                for boolvar, boolval in bool_assigns:
                    if boolval:
                        atomics.append(boolvar)
                    else:
                        atomics.append(f"!{boolvar}")
                if firstline:
                    line = ""
                else:
                    line = "else "
                # only need another if statement when we have more than one boolean variables
                if firstline or len(used_boolvars) > 1:
                    line += f"if({' && '.join(atomics)})"
                line += "\n    "
                line += f"{var} {op} {self.translate_expression(simp_expr)};"
                codelines.append(line)
                firstline = False
            code = "\n".join(codelines)
        else:
            if op == ":=":
                decl = f"{self.c_data_type(statement.dtype)} "
                op = "="
                if statement.constant:
                    decl = f"const {decl}"
            else:
                decl = ""
            code = f"{decl + var} {op} {self.translate_expression(expr)};"
        if len(comment):
            code += f" // {comment}"
        return code

    def translate_to_read_arrays(self, read, write, indices):
        lines = []
        # index and read arrays (index arrays first)
        for varname in itertools.chain(sorted(indices), sorted(read)):
            index_var = self.variable_indices[varname]
            var = self.variables[varname]
            if varname not in write:
                line = "const "
            else:
                line = ""
            line = f"{line + self.c_data_type(var.dtype)} {varname} = "
            line = f"{line + self.get_array_name(var)}[{index_var}];"
            lines.append(line)
        return lines

    def translate_to_declarations(self, read, write, indices):
        lines = []
        # simply declare variables that will be written but not read
        for varname in sorted(write):
            if varname not in read and varname not in indices:
                var = self.variables[varname]
                line = f"{self.c_data_type(var.dtype)} {varname};"
                lines.append(line)
        return lines

    def translate_to_statements(self, statements, conditional_write_vars):
        lines = []
        # the actual code
        for stmt in statements:
            line = self.translate_statement(stmt)
            if stmt.var in conditional_write_vars:
                condvar = conditional_write_vars[stmt.var]
                lines.append(f"if({condvar})")
                lines.append(f"    {line}")
            else:
                lines.append(line)
        return lines

    def translate_to_write_arrays(self, write):
        lines = []
        # write arrays
        for varname in sorted(write):
            index_var = self.variable_indices[varname]
            var = self.variables[varname]
            line = f"{self.get_array_name(var)}[{index_var}] = {varname};"
            lines.append(line)
        return lines

    def translate_one_statement_sequence(self, statements, scalar=False):
        # Note that we do not call this function from
        # `translate_statement_sequence` (which has been overwritten)
        # It is nevertheless implemented, so that it can be called explicitly
        # (e.g. from the GSL code generation)
        read, write, indices, cond_write = self.arrays_helper(statements)
        lines = []
        # index and read arrays (index arrays first)
        lines += self.translate_to_read_arrays(read, write, indices)
        # simply declare variables that will be written but not read
        lines += self.translate_to_declarations(read, write, indices)
        # the actual code
        lines += self.translate_to_statements(statements, cond_write)
        # write arrays
        lines += self.translate_to_write_arrays(write)
        return stripped_deindented_lines("\n".join(lines))

    def translate_statement_sequence(self, sc_statements, ve_statements):
        # This function is overwritten, since we do not want to completely
        # separate the code generation for scalar and vector code

        assert set(sc_statements.keys()) == set(ve_statements.keys())

        kwds = self.determine_keywords()

        sc_code = {}
        ve_code = {}

        for block_name in sc_statements:
            sc_block = sc_statements[block_name]
            ve_block = ve_statements[block_name]
            (sc_read, sc_write, sc_indices, sc_cond_write) = self.arrays_helper(
                sc_block
            )
            (ve_read, ve_write, ve_indices, ve_cond_write) = self.arrays_helper(
                ve_block
            )
            # We want to read all scalar variables that are needed in the
            # vector code already in the scalar code, if they are not written
            for varname in set(ve_read):
                var = self.variables[varname]
                if var.scalar and varname not in ve_write:
                    sc_read.add(varname)
                    ve_read.remove(varname)

            for code, stmts, read, write, indices, cond_write in [
                (sc_code, sc_block, sc_read, sc_write, sc_indices, sc_cond_write),
                (ve_code, ve_block, ve_read, ve_write, ve_indices, ve_cond_write),
            ]:
                lines = []
                # index and read arrays (index arrays first)
                lines += self.translate_to_read_arrays(read, write, indices)
                # simply declare variables that will be written but not read
                lines += self.translate_to_declarations(read, write, indices)
                # the actual code
                lines += self.translate_to_statements(stmts, cond_write)
                # write arrays
                lines += self.translate_to_write_arrays(write)
                code[block_name] = stripped_deindented_lines("\n".join(lines))

        return sc_code, ve_code, kwds

    def denormals_to_zero_code(self):
        if self.flush_denormals:
            return """
            #define CSR_FLUSH_TO_ZERO         (1 << 15)
            unsigned csr = __builtin_ia32_stmxcsr();
            csr |= CSR_FLUSH_TO_ZERO;
            __builtin_ia32_ldmxcsr(csr);
            """
        else:
            return ""

    def _add_user_function(self, varname, variable, added):
        impl = variable.implementations[self.codeobj_class]
        if (impl.name, variable) in added:
            return  # nothing to do
        else:
            added.add((impl.name, variable))
        support_code = []
        hash_defines = []
        pointers = []
        user_functions = [(varname, variable)]
        funccode = impl.get_code(self.owner)
        if isinstance(funccode, str):
            # Rename references to any dependencies if necessary
            for dep_name, dep in impl.dependencies.items():
                dep_impl = dep.implementations[self.codeobj_class]
                dep_impl_name = dep_impl.name
                if dep_impl_name is None:
                    dep_impl_name = dep.pyfunc.__name__
                if dep_name != dep_impl_name:
                    funccode = word_substitute(funccode, {dep_name: dep_impl_name})
            funccode = {"support_code": funccode}
        if funccode is not None:
            # To make namespace variables available to functions, we
            # create global variables and assign to them in the main
            # code
            func_namespace = impl.get_namespace(self.owner) or {}
            for ns_key, ns_value in func_namespace.items():
                if hasattr(ns_value, "dtype"):
                    if ns_value.shape == ():
                        raise NotImplementedError(
                            "Directly replace scalar values in the function "
                            "instead of providing them via the namespace"
                        )
                    type_str = f"{self.c_data_type(ns_value.dtype)}*"
                else:  # e.g. a function
                    type_str = "py::object"
                support_code.append(f"static {type_str} _namespace{ns_key};")
                pointers.append(f"_namespace{ns_key} = {ns_key};")
            support_code.append(deindent(funccode.get("support_code", "")))
            hash_defines.append(deindent(funccode.get("hashdefine_code", "")))

        dep_hash_defines = []
        dep_pointers = []
        dep_support_code = []
        if impl.dependencies is not None:
            for dep_name, dep in impl.dependencies.items():
                if dep_name not in self.variables:
                    self.variables[dep_name] = dep
                    dep_impl = dep.implementations[self.codeobj_class]
                    if dep_name != dep_impl.name:
                        self.func_name_replacements[dep_name] = dep_impl.name
                    user_function = self._add_user_function(dep_name, dep, added)
                    if user_function is not None:
                        hd, ps, sc, uf = user_function
                        dep_hash_defines.extend(hd)
                        dep_pointers.extend(ps)
                        dep_support_code.extend(sc)
                        user_functions.extend(uf)

        return (
            dep_hash_defines + hash_defines,
            dep_pointers + pointers,
            dep_support_code + support_code,
            user_functions,
        )

    def determine_keywords(self):
        # set up the restricted pointers, these are used so that the compiler
        # knows there is no aliasing in the pointers, for optimisation
        pointers = []
        # It is possible that several different variable names refer to the
        # same array. E.g. in gapjunction code, v_pre and v_post refer to the
        # same array if a group is connected to itself
        handled_pointers = set()
        template_kwds = {}
        # Again, do the import here to avoid a circular dependency.
        from brian2.devices.device import get_device

        device = get_device()
        for var in self.variables.values():
            if isinstance(var, ArrayVariable):
                # This is the "true" array name, not the restricted pointer.
                array_name = device.get_array_name(var)
                pointer_name = self.get_array_name(var)
                if pointer_name in handled_pointers:
                    continue
                if getattr(var, "ndim", 1) > 1:
                    continue  # multidimensional (dynamic) arrays have to be treated differently
                restrict = self.restrict
                # turn off restricted pointers for scalars for safety
                if var.scalar or var.size == 1:
                    restrict = " "
                line = (
                    f"{self.c_data_type(var.dtype)}* {restrict} {pointer_name} ="
                    f" {array_name};"
                )
                pointers.append(line)
                handled_pointers.add(pointer_name)

        # set up the functions
        user_functions = []
        support_code = []
        hash_defines = []
        added = set()  # keep track of functions that were added
        for varname, variable in list(self.variables.items()):
            if isinstance(variable, Function):
                user_func = self._add_user_function(varname, variable, added)
                if user_func is not None:
                    hd, ps, sc, uf = user_func
                    user_functions.extend(uf)
                    support_code.extend(sc)
                    pointers.extend(ps)
                    hash_defines.extend(hd)
        support_code.append(self.universal_support_code)

        keywords = {
            "pointers_lines": stripped_deindented_lines("\n".join(pointers)),
            "support_code_lines": stripped_deindented_lines("\n".join(support_code)),
            "hashdefine_lines": stripped_deindented_lines("\n".join(hash_defines)),
            "denormals_code_lines": stripped_deindented_lines(
                "\n".join(self.denormals_to_zero_code())
            ),
        }
        keywords.update(template_kwds)
        return keywords


################################################################################
# Implement functions
################################################################################
# Functions that exist under the same name in C++
for func in [
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "exp",
    "log",
    "log10",
    "sqrt",
    "ceil",
    "floor",
]:
    DEFAULT_FUNCTIONS[func].implementations.add_implementation(
        CPPCodeGenerator, code=None
    )
DEFAULT_FUNCTIONS["expm1"].implementations.add_implementation(
    CPPCodeGenerator, code=None, availability_check=C99Check("expm1")
)
DEFAULT_FUNCTIONS["log1p"].implementations.add_implementation(
    CPPCodeGenerator, code=None, availability_check=C99Check("log1p")
)

# Functions that need a name translation
for func, func_cpp in [
    ("arcsin", "asin"),
    ("arccos", "acos"),
    ("arctan", "atan"),
    ("int", "int_"),  # from stdint_compat.h
]:
    DEFAULT_FUNCTIONS[func].implementations.add_implementation(
        CPPCodeGenerator, code=None, name=func_cpp
    )

exprel_code = """
static inline double _exprel(double x)
{
    if (fabs(x) < 1e-16)
        return 1.0;
    if (x > 717)
        return INFINITY;
    return expm1(x)/x;
}
"""
DEFAULT_FUNCTIONS["exprel"].implementations.add_implementation(
    CPPCodeGenerator,
    code=exprel_code,
    name="_exprel",
    availability_check=C99Check("exprel"),
)

abs_code = """
#define _brian_abs std::abs
"""
DEFAULT_FUNCTIONS["abs"].implementations.add_implementation(
    CPPCodeGenerator, code=abs_code, name="_brian_abs"
)

clip_code = """
        template <typename T>
        static inline T _clip(const T value, const double a_min, const double a_max)
        {
            if (value < a_min)
                return a_min;
            if (value > a_max)
                return a_max;
            return value;
        }
        """
DEFAULT_FUNCTIONS["clip"].implementations.add_implementation(
    CPPCodeGenerator, code=clip_code, name="_clip"
)

sign_code = """
        template <typename T> static inline int _sign(T val) {
            return (T(0) < val) - (val < T(0));
        }
        """
DEFAULT_FUNCTIONS["sign"].implementations.add_implementation(
    CPPCodeGenerator, code=sign_code, name="_sign"
)

timestep_code = """
static inline int64_t _timestep(double t, double dt)
{
    return (int64_t)((t + 1e-3*dt)/dt);
}
"""
DEFAULT_FUNCTIONS["timestep"].implementations.add_implementation(
    CPPCodeGenerator, code=timestep_code, name="_timestep"
)

poisson_code = """
double _loggam(double x) {
  double x0, x2, xp, gl, gl0;
  int32_t k, n;

  static double a[10] = {8.333333333333333e-02, -2.777777777777778e-03,
                         7.936507936507937e-04, -5.952380952380952e-04,
                         8.417508417508418e-04, -1.917526917526918e-03,
                         6.410256410256410e-03, -2.955065359477124e-02,
                         1.796443723688307e-01, -1.39243221690590e+00};
  x0 = x;
  n = 0;
  if ((x == 1.0) || (x == 2.0))
    return 0.0;
  else if (x <= 7.0) {
    n = (int32_t)(7 - x);
    x0 = x + n;
  }
  x2 = 1.0 / (x0 * x0);
  xp = 2 * M_PI;
  gl0 = a[9];
  for (k=8; k>=0; k--) {
    gl0 *= x2;
    gl0 += a[k];
  }
  gl = gl0 / x0 + 0.5 * log(xp) + (x0 - 0.5) * log(x0) - x0;
  if (x <= 7.0) {
    for (k=1; k<=n; k++) {
      gl -= log(x0 - 1.0);
      x0 -= 1.0;
    }
  }
  return gl;
}

int32_t _poisson_mult(double lam, int _vectorisation_idx) {
  int32_t X;
  double prod, U, enlam;

  enlam = exp(-lam);
  X = 0;
  prod = 1.0;
  while (1) {
    U = _rand(_vectorisation_idx);
    prod *= U;
    if (prod > enlam)
      X += 1;
    else
      return X;
  }
}

int32_t _poisson_ptrs(double lam, int _vectorisation_idx) {
  int32_t k;
  double U, V, slam, loglam, a, b, invalpha, vr, us;

  slam = sqrt(lam);
  loglam = log(lam);
  b = 0.931 + 2.53 * slam;
  a = -0.059 + 0.02483 * b;
  invalpha = 1.1239 + 1.1328 / (b - 3.4);
  vr = 0.9277 - 3.6224 / (b - 2);

  while (1) {
    U = _rand(_vectorisation_idx) - 0.5;
    V = _rand(_vectorisation_idx);
    us = 0.5 - abs(U);
    k = (int32_t)floor((2 * a / us + b) * U + lam + 0.43);
    if ((us >= 0.07) && (V <= vr))
      return k;
    if ((k < 0) || ((us < 0.013) && (V > us)))
      continue;
    if ((log(V) + log(invalpha) - log(a / (us * us) + b)) <=
        (-lam + k * loglam - _loggam(k + 1)))
      return k;
  }
}
int32_t _poisson(double lam, int32_t _idx) {
  if (lam >= 10)
    return _poisson_ptrs(lam, _idx);
  else if (lam == 0)
    return 0;
  else
    return _poisson_mult(lam, _idx);
}
"""

DEFAULT_FUNCTIONS["poisson"].implementations.add_implementation(
    CPPCodeGenerator,
    code=poisson_code,
    name="_poisson",
    dependencies={"_rand": DEFAULT_FUNCTIONS["rand"]},
)
