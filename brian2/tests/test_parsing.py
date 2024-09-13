"""
Tests the brian2.parsing package
"""

from collections import namedtuple

import numpy as np
import pytest

from brian2 import Function
from brian2.codegen.generators.cpp_generator import CPPCodeGenerator
from brian2.core.functions import DEFAULT_FUNCTIONS
from brian2.core.preferences import prefs
from brian2.core.variables import Constant
from brian2.groups.group import Group
from brian2.parsing.dependencies import abstract_code_dependencies
from brian2.parsing.expressions import (
    _get_value_from_expression,
    is_boolean_expression,
    parse_expression_dimensions,
)
from brian2.parsing.functions import (
    abstract_code_from_function,
    extract_abstract_code_functions,
    substitute_abstract_code_functions,
)
from brian2.parsing.rendering import CPPNodeRenderer, NodeRenderer, NumpyNodeRenderer
from brian2.parsing.sympytools import str_to_sympy, sympy_to_str
from brian2.tests.utils import assert_allclose
from brian2.units import (
    DimensionMismatchError,
    Unit,
    amp,
    get_unit,
    have_same_dimensions,
    volt,
)
from brian2.units.fundamentalunits import DIMENSIONLESS, Dimension
from brian2.utils.logger import std_silent
from brian2.utils.stringtools import deindent, get_identifiers


# a simple Group for testing
class SimpleGroup(Group):
    def __init__(self, variables, namespace=None):
        self.variables = variables
        self.namespace = namespace


TEST_EXPRESSIONS = """
    a+b+c*d-f+g-(b+d)-(a-c)
    a**b**2
    a**(b**2)
    (a**b)**2
    a*(b+c*(a+b)*(a-(c*d)))
    a/b/c-a/(b/c)
    10//n
    n//10
    n//m
    10/n
    10.0/n
    n/10
    n/10.0
    n/m
    a<b
    a<=b
    a>b
    a>=b
    a==b
    a!=b
    a+1
    1+a
    1+3
    a>0.5 and b>0.5
    a>0.5 and b>0.5 or c>0.5
    a>0.5 and b>0.5 or not c>0.5
    2%4
    -1%4
    2.3%5.6
    2.3%5
    -1.2%3.4
    17e-12
    42e17
    """


def parse_expressions(renderer, evaluator, numvalues=10):
    exprs = [
        ([m for m in get_identifiers(l) if len(m) == 1], [], l.strip())
        for l in TEST_EXPRESSIONS.split("\n")
        if l.strip()
    ]
    i, imod = 1, 33
    for varids, funcids, expr in exprs:
        pexpr = renderer.render_expr(expr)
        n = 0
        for _ in range(numvalues):
            # assign some random values
            ns = {}
            for v in varids:
                if v in ["n", "m"]:  # integer values
                    ns[v] = i
                else:
                    ns[v] = float(i) / imod
                i = i % imod + 1
            r1 = eval(expr.replace("&", " and ").replace("|", " or "), ns)
            n += 1
            r2 = evaluator(pexpr, ns)
            try:
                # Use all close because we can introduce small numerical
                # difference through sympy's rearrangements
                assert_allclose(r1, r2, atol=1e-8)
            except AssertionError as e:
                raise AssertionError(
                    f"In expression {str(expr)} translated to {str(pexpr)} {str(e)}"
                )


def numpy_evaluator(expr, userns):
    ns = {}
    # exec 'from numpy import logical_not' in ns
    ns["logical_not"] = np.logical_not
    ns.update(**userns)
    for k in userns:
        if not k.startswith("_"):
            ns[k] = np.array([userns[k]])
    try:
        x = eval(expr, ns)
    except Exception as e:
        raise ValueError(
            f"Could not evaluate numpy expression {expr} exception {str(e)}"
        )
    if isinstance(x, np.ndarray):
        return x[0]
    else:
        return x


@pytest.mark.codegen_independent
def test_parse_expressions_python():
    parse_expressions(NodeRenderer(), eval)


@pytest.mark.codegen_independent
def test_parse_expressions_numpy():
    parse_expressions(NumpyNodeRenderer(), numpy_evaluator)


@pytest.mark.codegen_independent
def test_parse_expressions_sympy():
    # sympy is about symbolic calculation, the string returned by the renderer
    # contains "Symbol('a')" etc. so we cannot simply evaluate it in a
    # namespace.
    # We therefore use a different approach: Convert the expression to a
    # sympy expression via str_to_sympy (uses the SympyNodeRenderer internally),
    # then convert it back to a string via sympy_to_str and evaluate it

    class SympyRenderer:
        def render_expr(self, expr):
            return str_to_sympy(expr)

    def evaluator(expr, ns):
        expr = sympy_to_str(expr)
        ns = dict(ns)
        # Add the floor function which is used to implement floor division
        ns["floor"] = DEFAULT_FUNCTIONS["floor"]
        return eval(expr, ns)

    parse_expressions(SympyRenderer(), evaluator)


@pytest.mark.codegen_independent
def test_abstract_code_dependencies():
    code = """
    a = b+c
    d = b+c
    a = func_a()
    a = func_b()
    a = x+d
    """
    known_vars = {"a", "b", "c"}
    known_funcs = {"func_a"}
    res = abstract_code_dependencies(code, known_vars, known_funcs)
    expected_res = dict(
        all=[
            "a",
            "b",
            "c",
            "d",
            "x",
            "func_a",
            "func_b",
        ],
        read=["b", "c", "d", "x"],
        write=["a", "d"],
        funcs=["func_a", "func_b"],
        known_all=["a", "b", "c", "func_a"],
        known_read=["b", "c"],
        known_write=["a"],
        known_funcs=["func_a"],
        unknown_read=["d", "x"],
        unknown_write=["d"],
        unknown_funcs=["func_b"],
        undefined_read=["x"],
        newly_defined=["d"],
    )
    for k, v in expected_res.items():
        if not getattr(res, k) == set(v):
            raise AssertionError(
                f"For '{k}' result is {getattr(res, k)} expected {set(v)}"
            )


@pytest.mark.codegen_independent
def test_is_boolean_expression():
    # dummy "Variable" class
    Var = namedtuple("Var", ["is_boolean"])

    # dummy function object
    class Func:
        def __init__(self, returns_bool=False):
            self._returns_bool = returns_bool

    # variables / functions
    a = Constant("a", value=True)
    b = Constant("b", value=False)
    c = Constant("c", value=5)
    f = Func(returns_bool=True)
    g = Func(returns_bool=False)
    s1 = Var(is_boolean=True)
    s2 = Var(is_boolean=False)

    variables = {"a": a, "b": b, "c": c, "f": f, "g": g, "s1": s1, "s2": s2}

    EVF = [
        (True, "a or b"),
        (False, "c"),
        (False, "s2"),
        (False, "g(s1)"),
        (True, "s2 > c"),
        (True, "c > 5"),
        (True, "True"),
        (True, "a<b"),
        (True, "not (a>=b)"),
        (False, "a+b"),
        (True, "f(c)"),
        (False, "g(c)"),
        (
            True,
            "f(c) or a<b and s1",
        ),
    ]
    for expect, expr in EVF:
        ret_val = is_boolean_expression(expr, variables)
        if expect != ret_val:
            raise AssertionError(
                f"is_boolean_expression({expr!r}) returned 'ret_val', "
                "but was supposed to return 'expect'."
            )
    with pytest.raises(SyntaxError):
        is_boolean_expression("a<b and c", variables)
    with pytest.raises(SyntaxError):
        is_boolean_expression("a or foo", variables)
    with pytest.raises(SyntaxError):
        is_boolean_expression("ot a", variables)  # typo
    with pytest.raises(SyntaxError):
        is_boolean_expression("g(c) and f(a)", variables)


@pytest.mark.codegen_independent
@pytest.mark.parametrize(
    "expect,expr",
    [
        (volt * amp, "a+b*c"),
        (DimensionMismatchError, "a+b"),
        (DimensionMismatchError, "a<b"),
        (1, "a<b*c"),
        (1, "a or b"),
        (1, "not (a >= b*c)"),
        (DimensionMismatchError, "a or b<c"),
        (1, "a/(b*c)<1"),
        (1, "a/(a-a)"),
        (1, "a<mV*mA"),
        (volt**2, "b**2"),
        (volt * amp, "a%(b*c)"),
        (volt, "-b"),
        (1, "(a/a)**(a/a)"),
        # Expressions involving functions
        (volt, "rand()*b"),
        (volt**0.5, "sqrt(b)"),
        (volt, "ceil(b)"),
        (volt, "sqrt(randn()*b**2)"),
        (1, "sin(b/b)"),
        (DimensionMismatchError, "sin(b)"),
        (DimensionMismatchError, "sqrt(b) + b"),
        (SyntaxError, "sqrt(b, b)"),
        (SyntaxError, "sqrt()"),
        (SyntaxError, "int(1, 2)"),
    ],
)
def test_parse_expression_unit(expect, expr):
    Var = namedtuple("Var", ["dim", "dtype"])
    variables = {
        "a": Var(dim=(volt * amp).dim, dtype=np.float64),
        "b": Var(dim=volt.dim, dtype=np.float64),
        "c": Var(dim=amp.dim, dtype=np.float64),
    }
    group = SimpleGroup(namespace={}, variables=variables)
    all_variables = {}
    for name in get_identifiers(expr):
        if name in variables:
            all_variables[name] = variables[name]
        else:
            all_variables[name] = group._resolve(name, {})

    if isinstance(expect, type) and issubclass(expect, Exception):
        with pytest.raises(expect):
            parse_expression_dimensions(expr, all_variables)
    else:
        u = parse_expression_dimensions(expr, all_variables)
        assert have_same_dimensions(u, expect)


@pytest.mark.codegen_independent
@pytest.mark.parametrize("expr", ["a**b", "a << b", "int(True"])  # typo
def test_parse_expression_unit_wrong_expressions(expr):
    Var = namedtuple("Var", ["dim", "dtype"])
    variables = {
        "a": Var(dim=(volt * amp).dim, dtype=np.float64),
        "b": Var(dim=volt.dim, dtype=np.float64),
        "c": Var(dim=amp.dim, dtype=np.float64),
    }
    all_variables = {}
    group = SimpleGroup(namespace={}, variables=variables)
    for name in get_identifiers(expr):
        if name in variables:
            all_variables[name] = variables[name]
        else:
            all_variables[name] = group._resolve(name, {})
    with pytest.raises(SyntaxError):
        parse_expression_dimensions(expr, all_variables)


@pytest.mark.codegen_independent
@pytest.mark.parametrize(
    "expr,correct",
    [
        ("sin(5)", True),
        ("3 + sin(5)", True),
        ("d + sin(5)", True),
        ("3 + sin(d)", True),
        ("sin(5*mV)", False),
        ("sin(a)", False),
        ("3*mV + sin(5)", False),
        ("b + sin(5)", False),
        ("sqrt(b**2) + sqrt(7*mV**2)", True),
        ("sqrt(d) + sqrt(7*mV**2)", False),
        ("b + clip(3*mV, 0*mV, 5*mV)", True),
        ("b + clip(3*mV, 0, 5)", False),
        ("b + foo(7, 3*mV, 9)", True),
        ("a + foo(7*nA, 3*mV, 9*nA)", True),
        ("b + foo(7, 3*mV, 9*mV)", False),
        ("a + foo(7*nA, 3*mV, 9)", False),
    ],
)
def test_parse_expression_unit_functions(expr, correct):
    Var = namedtuple("Var", ["dim", "dtype"])

    def foo(x, y, z):
        return (x + z) * y

    variables = {
        "a": Var(dim=(volt * amp).dim, dtype=np.float64),
        "b": Var(dim=volt.dim, dtype=np.float64),
        "c": Var(dim=amp.dim, dtype=np.float64),
        "d": Var(dim=DIMENSIONLESS, dtype=np.float64),
        "foo": Function(
            pyfunc=foo,
            arg_units=[None, volt, "x"],
            arg_names=["x", "y", "z"],
            return_unit=lambda x, y, z: x * y,
        ),
    }
    all_variables = {}
    group = SimpleGroup(namespace={}, variables=variables)
    for name in get_identifiers(expr):
        if name in variables:
            all_variables[name] = variables[name]
        else:
            all_variables[name] = group._resolve(name, {})
    if correct:
        assert isinstance(parse_expression_dimensions(expr, all_variables), Dimension)
    else:
        with pytest.raises(DimensionMismatchError):
            parse_expression_dimensions(expr, all_variables)


@pytest.mark.codegen_independent
def test_value_from_expression():
    # This function is used to get the value of an exponent, necessary for unit checking

    constants = {"c": 3}

    # dummy class
    class C:
        pass

    variables = {"s_constant_scalar": C(), "s_non_constant": C(), "s_non_scalar": C()}
    variables["s_constant_scalar"].scalar = True
    variables["s_constant_scalar"].constant = True
    variables["s_constant_scalar"].get_value = lambda: 2.0
    variables["s_non_scalar"].constant = True
    variables["s_non_constant"].scalar = True
    variables["c"] = Constant("c", value=3)

    expressions = [
        "1",
        "-0.5",
        "c",
        "2**c",
        "(c + 3) * 5",
        "c + s_constant_scalar",
        "True",
        "False",
    ]

    for expr in expressions:
        eval_expr = expr.replace("s_constant_scalar", "s_constant_scalar.get_value()")
        assert float(
            eval(eval_expr, variables, constants)
        ) == _get_value_from_expression(expr, variables)

    wrong_expressions = ["s_non_constant", "s_non_scalar", "c or True"]
    for expr in wrong_expressions:
        with pytest.raises(SyntaxError):
            _get_value_from_expression(expr, variables)


@pytest.mark.codegen_independent
def test_abstract_code_from_function():
    # test basic functioning
    def f(x):
        y = x + 1
        return y * y

    ac = abstract_code_from_function(f)
    assert ac.name == "f"
    assert ac.args == ["x"]
    assert ac.code.strip() == "y = x + 1"
    assert ac.return_expr == "y * y"
    # Check that unsupported features raise an error

    def f(x):
        return x[:]

    with pytest.raises(SyntaxError):
        abstract_code_from_function(f)

    def f(x, **kwarg):
        return x

    with pytest.raises(SyntaxError):
        abstract_code_from_function(f)

    def f(x, *args):
        return x

    with pytest.raises(SyntaxError):
        abstract_code_from_function(f)


@pytest.mark.codegen_independent
def test_extract_abstract_code_functions():
    code = """
    def f(x):
        return x*x
        
    def g(V):
        V += 1
        
    irrelevant_code_here()
    """
    funcs = extract_abstract_code_functions(code)
    assert funcs["f"].return_expr == "x * x"
    assert funcs["g"].args == ["V"]


@pytest.mark.codegen_independent
def test_substitute_abstract_code_functions():
    def f(x):
        y = x * x
        return y

    def g(x):
        return f(x) + 1

    code = """
    z = f(x)
    z = f(x)+f(y)
    w = f(z)
    h = f(f(w))
    p = g(g(x))
    """
    funcs = [
        abstract_code_from_function(f),
        abstract_code_from_function(g),
    ]
    subcode = substitute_abstract_code_functions(code, funcs)
    for x, y in [(0, 1), (1, 0), (0.124323, 0.4549483)]:
        ns1 = {"x": x, "y": y, "f": f, "g": g}
        ns2 = {"x": x, "y": y}
        exec(deindent(code), ns1)
        exec(subcode, ns2)
        for k in ["z", "w", "h", "p"]:
            assert ns1[k] == ns2[k]


@pytest.mark.codegen_independent
def test_sympytools():
    # sympy_to_str(str_to_sympy(x)) should equal x

    # Note that the test below is quite fragile since sympy might rearrange the
    # order of symbols
    expressions = [
        "randn()",  # argumentless function
        "x + sin(pi*freq*t)",  # expression with a constant
        "c * userfun(t + x)",  # non-sympy function
        "abs(x) + ceil(y)",  # functions with a different name in sympy
        "inf",  # constant with a different name in sympy
        "not(b)",  # boolean expression
    ]

    for expr in expressions:
        expr2 = sympy_to_str(str_to_sympy(expr))
        assert expr.replace(" ", "") == expr2.replace(" ", ""), f"{expr} != {expr2}"


@pytest.mark.codegen_independent
def test_error_messages():
    nr = NodeRenderer()
    expr_expected = [
        ("3^2", "^", "**"),
        ("int(not_refractory | (v > 30))", "|", "or"),
        ("int((v > 30) & (w < 20))", "&", "and"),
        ("x +* 3", "", ""),
        ("v[index]", "indexing", ""),
        ("v.value", "attribute", ""),
        ("(v, w)", "tuple", ""),
    ]
    for expr, expected_1, expected_2 in expr_expected:
        try:
            nr.render_expr(expr)
            raise AssertionError(f"Excepted {expr} to raise a SyntaxError.")
        except SyntaxError as exc:
            message = str(exc)
            assert expected_1 in message
            assert expected_2 in message


@pytest.mark.codegen_independent
def test_sympy_infinity():
    # See github issue #1061
    assert sympy_to_str(str_to_sympy("inf")) == "inf"
    assert sympy_to_str(str_to_sympy("-inf")) == "-inf"


if __name__ == "__main__":
    from _pytest.outcomes import Skipped

    test_parse_expressions_python()
    test_parse_expressions_numpy()
    try:
        test_parse_expressions_cpp()
    except Skipped:
        pass
    test_parse_expressions_sympy()
    test_abstract_code_dependencies()
    test_is_boolean_expression()
    test_parse_expression_unit()
    test_value_from_expression()
    test_abstract_code_from_function()
    test_extract_abstract_code_functions()
    test_substitute_abstract_code_functions()
    test_sympytools()
    test_error_messages()
    test_sympy_infinity()
