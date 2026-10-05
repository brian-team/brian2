"""
Unit tests for function-related logic that does not require a full simulation.

These tests exercise pure Python logic — constant values, code renderer
special cases, the ``declare_types`` decorator, and the ``Function`` API —
without constructing any NeuronGroup or calling ``run()``.

Mocking is used where a dependency would normally require heavy
infrastructure (e.g. a full ArrayVariable needs a device, owner and size).
A MagicMock stands in as a lightweight fake that satisfies the isinstance
checks in the code under test without building anything real.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.testing import assert_allclose

from brian2 import e, inf, pi
from brian2.codegen.permutation_analysis import (
    OrderDependenceError,
    check_for_order_independence,
)
from brian2.codegen.statements import Statement
from brian2.core.functions import Function, declare_types
from brian2.parsing.rendering import CPPNodeRenderer, NumpyNodeRenderer
from brian2.parsing.sympytools import check_expression_for_multiple_stateful_functions
from brian2.units import amp, volt, watt
from brian2.units.fundamentalunits import DimensionMismatchError


# ---------------------------------------------------------------------------
# Symbolic constants
# ---------------------------------------------------------------------------


@pytest.mark.codegen_independent
def test_constants_values():
    """Symbolic constants pi, e and inf have correct float values."""
    assert_allclose(float(pi), np.pi)
    assert_allclose(float(e), np.e)
    assert float(inf) == np.inf


# ---------------------------------------------------------------------------
# CPPNodeRenderer special cases
#
# NumpyNodeRenderer passes most names through as-is (constants like ``pi``
# and ``e`` live in the execution namespace, not the code string).
# CPPNodeRenderer has explicit substitutions for a handful of cases.
# ---------------------------------------------------------------------------


@pytest.mark.codegen_independent
def test_cpp_renderer_inf():
    """CPPNodeRenderer maps ``inf`` to the C macro ``INFINITY``."""
    renderer = CPPNodeRenderer()
    assert renderer.render_expr("inf") == "INFINITY"


@pytest.mark.codegen_independent
def test_cpp_renderer_boolean_literals():
    """CPPNodeRenderer maps Python True/False to C++ true/false."""
    renderer = CPPNodeRenderer()
    assert renderer.render_expr("True") == "true"
    assert renderer.render_expr("False") == "false"


@pytest.mark.codegen_independent
def test_cpp_renderer_power_operator():
    """CPPNodeRenderer rewrites ``x ** y`` as ``_brian_pow(x, y)``."""
    renderer = CPPNodeRenderer()
    result = renderer.render_expr("x ** y")
    assert result == "_brian_pow(x, y)"


@pytest.mark.codegen_independent
def test_cpp_renderer_mod_operator():
    """CPPNodeRenderer rewrites ``x % y`` as ``_brian_mod(x, y)``."""
    renderer = CPPNodeRenderer()
    result = renderer.render_expr("x % y")
    assert result == "_brian_mod(x, y)"


@pytest.mark.codegen_independent
def test_numpy_renderer_boolean_operators():
    """NumpyNodeRenderer uses element-wise ``&``/``|`` for and/or."""
    renderer = NumpyNodeRenderer()
    assert renderer.render_expr("a and b") == "a & b"
    assert renderer.render_expr("a or b") == "a | b"


@pytest.mark.codegen_independent
def test_numpy_renderer_not_operator():
    """NumpyNodeRenderer wraps ``not x`` as ``logical_not(x)``."""
    renderer = NumpyNodeRenderer()
    assert renderer.render_expr("not a") == "logical_not(a)"


# ---------------------------------------------------------------------------
# ``declare_types`` decorator
#
# Extracted from ``test_declare_types`` in test_functions.py.  The decorator
# operates purely on the function object and never needs a simulation.
# ---------------------------------------------------------------------------


@pytest.mark.codegen_independent
def test_declare_types_sets_attributes():
    """@declare_types sets _arg_types and _return_type on the function."""

    @declare_types(a="integer", b="float", result="highest")
    def f(a, b):
        return a * b

    assert f._arg_types == ["integer", "float"]
    assert f._return_type == "highest"


@pytest.mark.codegen_independent
def test_declare_types_defaults():
    """Unspecified argument types default to 'any'; result defaults to 'float'."""

    @declare_types(b="float")
    def f(a, b, c):
        return a * b * c

    assert f._arg_types == ["any", "float", "any"]
    assert f._return_type == "float"


@pytest.mark.codegen_independent
def test_declare_types_invalid_arg_type():
    """@declare_types raises ValueError for an unrecognised argument type."""

    def make_bad():
        @declare_types(b="floating")  # 'floating' is not a valid type
        def f(a, b, c):
            return a * b * c

    with pytest.raises(ValueError):
        make_bad()


@pytest.mark.codegen_independent
def test_declare_types_invalid_return_type():
    """@declare_types raises ValueError for an unrecognised return type."""

    def make_bad():
        @declare_types(result="double")  # 'double' is not a valid return type
        def f(a, b):
            return a * b

    with pytest.raises(ValueError):
        make_bad()


@pytest.mark.codegen_independent
def test_declare_types_unknown_argument_name():
    """@declare_types raises ValueError when a type is given for a non-existent argument."""

    def make_bad():
        @declare_types(d="float")  # 'd' is not an argument of f(a, b, c)
        def f(a, b, c):
            return a * b * c

    with pytest.raises(ValueError):
        make_bad()


# ---------------------------------------------------------------------------
# Function creation and validation
#
# Extracted from ``test_manual_user_defined_function`` in test_functions.py.
# The validation happens inside Function.__init__ — no simulation needed.
# ---------------------------------------------------------------------------


@pytest.mark.codegen_independent
def test_function_creation_missing_arg_units():
    """Function() raises ValueError when arg_units is not provided and the
    wrapped function has no @check_units decoration."""

    def foo(x, y):
        return x + y

    with pytest.raises(ValueError):
        Function(foo, return_unit=volt)


@pytest.mark.codegen_independent
def test_function_creation_missing_return_unit():
    """Function() raises ValueError when return_unit is not provided and the
    wrapped function has no @check_units decoration."""

    def foo(x, y):
        return x + y

    with pytest.raises(ValueError):
        Function(foo, arg_units=[volt, volt])


@pytest.mark.codegen_independent
def test_function_creation_string_unit_without_arg_names():
    """Function() raises TypeError when a string unit is used but arg_names
    is not provided."""

    def foo(x, y):
        return x + y

    with pytest.raises(TypeError):
        Function(foo, arg_units=[volt, "x"])


@pytest.mark.codegen_independent
def test_function_creation_mismatched_arg_names_length():
    """Function() raises TypeError when arg_names and arg_units have different
    lengths."""

    def foo(x, y):
        return x + y

    with pytest.raises(TypeError):
        Function(foo, arg_units=[volt, "x"], arg_names=["x"])  # needs two entries


# ---------------------------------------------------------------------------
# Function direct calls with unit checking
#
# Extracted from ``test_manual_user_defined_function`` in test_functions.py.
# Calling a Function object directly exercises unit-checking logic without
# any code generation.
# ---------------------------------------------------------------------------


@pytest.mark.codegen_independent
def test_function_direct_call_fixed_units():
    """A Function with fixed arg/return units can be called directly."""

    def foo(x, y):
        return x + y + 3 * volt

    foo = Function(foo, arg_units=[volt, volt], return_unit=volt)
    assert foo(1 * volt, 2 * volt) == 6 * volt


@pytest.mark.codegen_independent
def test_function_direct_call_dependent_units():
    """A Function with 'same-unit' argument constraints can be called directly."""

    def bar(a, b, c):
        return a * (b + c)

    bar = Function(
        bar,
        arg_units=[None, None, "b"],
        arg_names=["a", "b", "c"],
        return_unit=lambda a, b, c: a * b,
    )
    assert bar(2, 3 * volt, 5 * volt) == 16 * volt
    assert bar(2 * amp, 3 * volt, 5 * volt) == 16 * watt
    assert bar(2 * volt, 3 * amp, 5 * amp) == 16 * watt


@pytest.mark.codegen_independent
def test_function_direct_call_unit_mismatch_raises():
    """Calling a Function with incompatible units raises DimensionMismatchError."""

    def bar(a, b, c):
        return a * (b + c)

    bar = Function(
        bar,
        arg_units=[None, None, "b"],
        arg_names=["a", "b", "c"],
        return_unit=lambda a, b, c: a * b,
    )
    with pytest.raises(DimensionMismatchError):
        bar(2, 3 * volt, 5 * amp)  # b and c must have the same unit


# ---------------------------------------------------------------------------
# Multiple stateful function calls
#
# ``check_expression_for_multiple_stateful_functions`` detects expressions
# like ``rand() - rand()`` which are problematic because sympy might simplify
# them to 0.  The current integration test for this builds a full NeuronGroup
# and calls run(0*ms).  Here we call the parsing function directly.
# ---------------------------------------------------------------------------


def _make_stateful_function():
    """Helper: a minimal Function with stateless=False (like rand)."""

    def impl(_):
        return 0.5

    return Function(
        impl, arg_units=[], return_unit=1, stateless=False, auto_vectorise=True
    )


def _make_stateless_function():
    """Helper: a minimal Function with stateless=True (like sin)."""

    def impl(x):
        return np.sin(x)

    return Function(impl, arg_units=[1], return_unit=1, stateless=True)


@pytest.mark.codegen_independent
def test_multiple_stateful_calls_raises():
    """check_expression_for_multiple_stateful_functions raises NotImplementedError
    when a stateful function appears more than once in an expression."""
    rand = _make_stateful_function()
    with pytest.raises(NotImplementedError):
        check_expression_for_multiple_stateful_functions(
            "rand() - rand()", {"rand": rand}
        )


@pytest.mark.codegen_independent
def test_single_stateful_call_ok():
    """A single call to a stateful function does not raise."""
    rand = _make_stateful_function()
    # Should not raise — only one call
    check_expression_for_multiple_stateful_functions("rand()", {"rand": rand})


@pytest.mark.codegen_independent
def test_multiple_stateless_calls_ok():
    """Multiple calls to a stateless function (e.g. sin) are fine."""
    sin = _make_stateless_function()
    # sin is stateless so sin(v) + sin(v) is safe
    check_expression_for_multiple_stateful_functions("sin(v) + sin(v)", {"sin": sin})


# ---------------------------------------------------------------------------
# Order-dependence detection with mocking
#
# ``check_for_order_independence`` checks whether a sequence of statements
# can be executed in any order without changing the result.  It needs a
# ``variables`` dict containing the actual variable objects.  Building a real
# ``ArrayVariable`` requires a device, owner, and size — heavy infrastructure.
#
# Instead we use MagicMock() as a stand-in.  The function only calls
# isinstance(var, Function) and isinstance(var, Constant) on each variable;
# a plain MagicMock fails both checks (it has no spec), so it is treated
# as a regular array variable — exactly what we need.
# ---------------------------------------------------------------------------


@pytest.mark.codegen_independent
def test_stateful_function_raises_order_dependence_error():
    """check_for_order_independence raises OrderDependenceError when a
    stateful function is used — the array variable is a MagicMock."""
    rand = _make_stateful_function()

    # MagicMock stands in for a real ArrayVariable.
    # isinstance(MagicMock(), Function) → False
    # isinstance(MagicMock(), Constant) → False
    # so the function treats it as a plain array variable — no device needed.
    mock_v = MagicMock()

    stmt = Statement("v", "=", "rand()", "", float)
    variables = {"rand": rand, "v": mock_v}
    indices = {"v": "_idx"}

    with pytest.raises(OrderDependenceError):
        check_for_order_independence([stmt], variables, indices)


@pytest.mark.codegen_independent
def test_stateless_function_no_order_dependence():
    """check_for_order_independence does not raise for a stateless function."""
    sin = _make_stateless_function()
    mock_v = MagicMock()

    stmt = Statement("v", "=", "sin(v)", "", float)
    variables = {"sin": sin, "v": mock_v}
    indices = {"v": "_idx"}

    # Should not raise — sin is stateless and v only depends on itself
    check_for_order_independence([stmt], variables, indices)
