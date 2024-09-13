import json
import os
import platform
import socket
from collections import namedtuple

import numpy as np
import pytest

from brian2 import _cache_dirs_and_extensions, clear_cache, prefs
from brian2.codegen.codeobject import CodeObject
from brian2.codegen.cpp_prefs import compiler_supports_c99, get_compiler_and_args
from brian2.codegen.generators.cython_generator import CythonNodeRenderer
from brian2.codegen.optimisation import optimise_statements
from brian2.codegen.runtime.cython_rt import CythonCodeObject
from brian2.codegen.statements import Statement
from brian2.codegen.translation import (
    analyse_identifiers,
    get_identifiers_recursively,
    make_statements,
    parse_statement,
)
from brian2.core.functions import DEFAULT_CONSTANTS, DEFAULT_FUNCTIONS, Function
from brian2.core.variables import ArrayVariable, Constant, Subexpression, Variable
from brian2.devices.device import auto_target, device
from brian2.parsing.rendering import CPPNodeRenderer, NodeRenderer, NumpyNodeRenderer
from brian2.parsing.sympytools import str_to_sympy, sympy_to_str
from brian2.units import ms, second
from brian2.units.fundamentalunits import Unit
from brian2.utils.logger import catch_logs

FakeGroup = namedtuple("FakeGroup", ["variables"])


@pytest.mark.codegen_independent
def test_auto_target():
    # very basic test that the "auto" codegen target is useable
    assert issubclass(auto_target(), CodeObject)


@pytest.mark.codegen_independent
def test_analyse_identifiers():
    """
    Test that the analyse_identifiers function works on a simple clear example.
    """
    code = """
    a = b+c
    d = e+f
    """
    known = {
        "b": Variable(name="b"),
        "c": Variable(name="c"),
        "d": Variable(name="d"),
        "g": Variable(name="g"),
    }

    defined, used_known, dependent = analyse_identifiers(code, known)
    assert "a" in defined  # There might be an additional constant added by the
    # loop-invariant optimisation
    assert used_known == {"b", "c", "d"}
    assert dependent == {"e", "f"}


@pytest.mark.codegen_independent
def test_get_identifiers_recursively():
    """
    Test finding identifiers including subexpressions.
    """
    variables = {
        "sub1": Subexpression(
            name="sub1",
            dtype=np.float32,
            expr="sub2 * z",
            owner=FakeGroup(variables={}),
            device=None,
        ),
        "sub2": Subexpression(
            name="sub2",
            dtype=np.float32,
            expr="5 + y",
            owner=FakeGroup(variables={}),
            device=None,
        ),
        "x": Variable(name="x"),
    }
    identifiers = get_identifiers_recursively(["_x = sub1 + x"], variables)
    assert identifiers == {"x", "_x", "y", "z", "sub1", "sub2"}


@pytest.mark.codegen_independent
def test_write_to_subexpression():
    variables = {
        "a": Subexpression(
            name="a",
            dtype=np.float32,
            owner=FakeGroup(variables={}),
            device=None,
            expr="2*z",
        ),
        "z": Variable(name="z"),
    }

    # Writing to a subexpression is not allowed
    code = "a = z"
    with pytest.raises(SyntaxError):
        make_statements(code, variables, np.float32)


@pytest.mark.codegen_independent
def test_repeated_subexpressions():
    variables = {
        "a": Subexpression(
            name="a",
            dtype=np.float32,
            owner=FakeGroup(variables={}),
            device=None,
            expr="2*z",
        ),
        "x": Variable(name="x"),
        "y": Variable(name="y"),
        "z": Variable(name="z"),
    }
    # subexpression a (referring to z) is used twice, but can be reused the
    # second time (no change to z)
    code = """
    x = a
    y = a
    """
    scalar_stmts, vector_stmts = make_statements(code, variables, np.float32)
    assert len(scalar_stmts) == 0
    assert [stmt.var for stmt in vector_stmts] == ["a", "x", "y"]
    assert vector_stmts[0].constant

    code = """
    x = a
    z *= 2
    """
    scalar_stmts, vector_stmts = make_statements(code, variables, np.float32)
    assert len(scalar_stmts) == 0
    assert [stmt.var for stmt in vector_stmts] == ["a", "x", "z"]
    # Note that we currently do not mark the subexpression as constant in this
    # case, because its use after the "z *=2" line would actually redefine it.
    # Our algorithm is currently not smart enough to detect that it is actually
    # not used afterwards

    # a refers to z, therefore we have to redefine a after z changed, and a
    # cannot be constant
    code = """
    x = a
    z *= 2
    y = a
    """
    scalar_stmts, vector_stmts = make_statements(code, variables, np.float32)
    assert len(scalar_stmts) == 0
    assert [stmt.var for stmt in vector_stmts] == ["a", "x", "z", "a", "y"]
    assert not any(stmt.constant for stmt in vector_stmts)


@pytest.mark.codegen_independent
def test_nested_subexpressions():
    """
    This test checks that code translation works with nested subexpressions.
    """
    code = """
    x = a + b + c
    c = 1
    x = a + b + c
    d = 1
    x = a + b + c
    """
    variables = {
        "a": Subexpression(
            name="a",
            dtype=np.float32,
            owner=FakeGroup(variables={}),
            device=None,
            expr="b*b+d",
        ),
        "b": Subexpression(
            name="b",
            dtype=np.float32,
            owner=FakeGroup(variables={}),
            device=None,
            expr="c*c*c",
        ),
        "c": Variable(name="c"),
        "d": Variable(name="d"),
    }
    scalar_stmts, vector_stmts = make_statements(code, variables, np.float32)
    assert len(scalar_stmts) == 0
    evalorder = "".join(stmt.var for stmt in vector_stmts)
    # This is the order that variables ought to be evaluated in (note that
    # previously this test did not expect the last "b" evaluation, because its
    # value did not change (c was not changed). We have since removed this
    # subexpression caching, because it did not seem to apply in practical
    # use cases)
    assert evalorder == "baxcbaxdbax"


@pytest.mark.codegen_independent
def test_apply_loop_invariant_optimisation():
    variables = {
        "v": Variable("v", scalar=False),
        "w": Variable("w", scalar=False),
        "dt": Constant("dt", dimensions=second.dim, value=0.1 * ms),
        "tau": Constant("tau", dimensions=second.dim, value=10 * ms),
        "exp": DEFAULT_FUNCTIONS["exp"],
    }
    statements = [
        Statement("v", "=", "dt*w*exp(-dt/tau)/tau + v*exp(-dt/tau)", "", np.float32),
        Statement("w", "=", "w*exp(-dt/tau)", "", np.float32),
    ]
    scalar, vector = optimise_statements([], statements, variables)
    # The optimisation should pull out at least exp(-dt / tau)
    assert len(scalar) >= 1
    assert np.issubdtype(scalar[0].dtype, np.floating)
    assert scalar[0].var == "_lio_1"
    assert len(vector) == 2
    assert all("_lio_" in stmt.expr for stmt in vector)


@pytest.mark.codegen_independent
def test_apply_loop_invariant_optimisation_integer():
    variables = {
        "v": Variable("v", scalar=False),
        "N": Constant("N", 10),
        "b": Variable("b", scalar=True, dtype=int),
        "c": Variable("c", scalar=True, dtype=int),
        "d": Variable("d", scalar=True, dtype=int),
        "y": Variable("y", scalar=True, dtype=float),
        "z": Variable("z", scalar=True, dtype=float),
        "w": Variable("w", scalar=True, dtype=float),
    }
    statements = [
        Statement("v", "=", "v % (2*3*N)", "", np.float32),
        # integer version doesn't get rewritten but float version does
        Statement("a", ":=", "b//(c//d)", "", int),
        Statement("x", ":=", "y/(z/w)", "", float),
    ]
    scalar, vector = optimise_statements([], statements, variables)
    assert len(scalar) == 3
    assert np.issubdtype(scalar[0].dtype, np.signedinteger)
    assert scalar[0].var == "_lio_1"
    expr = scalar[0].expr.replace(" ", "")
    assert expr == "6*N" or expr == "N*6"
    assert np.issubdtype(scalar[1].dtype, np.signedinteger)
    assert scalar[1].var == "_lio_2"
    expr = scalar[1].expr.replace(" ", "")
    assert expr == "b//(c//d)"
    assert np.issubdtype(scalar[2].dtype, np.floating)
    assert scalar[2].var == "_lio_3"
    expr = scalar[2].expr.replace(" ", "")
    assert expr == "(y*w)/z" or expr == "(w*y)/z"


@pytest.mark.codegen_independent
def test_apply_loop_invariant_optimisation_boolean():
    variables = {
        "v1": Variable("v1", scalar=False),
        "v2": Variable("v2", scalar=False),
        "N": Constant("N", 10),
        "b": Variable("b", scalar=True, dtype=bool),
        "c": Variable("c", scalar=True, dtype=bool),
        "int": DEFAULT_FUNCTIONS["int"],
        "foo": Function(
            lambda x: None,
            arg_units=[Unit(1)],
            return_unit=Unit(1),
            arg_types=["boolean"],
            return_type="float",
            stateless=False,
        ),
    }
    # The calls for "foo" cannot be pulled out, since foo is marked as stateful
    statements = [
        Statement("v1", "=", "1.0*int(b and c)", "", np.float32),
        Statement("v1", "=", "1.0*foo(b and c)", "", np.float32),
        Statement("v2", "=", "int(not b and True)", "", np.float32),
        Statement("v2", "=", "foo(not b and True)", "", np.float32),
    ]
    scalar, vector = optimise_statements([], statements, variables)
    assert len(scalar) == 4
    assert scalar[0].expr == "1.0 * int(b and c)"
    assert scalar[1].expr == "b and c"
    assert scalar[2].expr == "int((not b) and True)"
    assert scalar[3].expr == "(not b) and True"
    assert len(vector) == 4
    assert vector[0].expr == "_lio_1"
    assert vector[1].expr == "foo(_lio_2)"
    assert vector[2].expr == "_lio_3"
    assert vector[3].expr == "foo(_lio_4)"


@pytest.mark.codegen_independent
def test_apply_loop_invariant_optimisation_no_optimisation():
    variables = {
        "v1": Variable("v1", scalar=False),
        "v2": Variable("v2", scalar=False),
        "N": Constant("N", 10),
        "s1": Variable("s1", scalar=True, dtype=float),
        "s2": Variable("s2", scalar=True, dtype=float),
        "rand": DEFAULT_FUNCTIONS["rand"],
    }
    statements = [
        # This should not be simplified to 0!
        Statement("v1", "=", "rand() - rand()", "", float),
        Statement("v1", "=", "3*rand() - 3*rand()", "", float),
        Statement("v1", "=", "3*rand() - ((1+2)*rand())", "", float),
        # This should not pull out rand()*N
        Statement("v1", "=", "s1*rand()*N", "", float),
        Statement("v1", "=", "s2*rand()*N", "", float),
        # This is not important mathematically, but it would change the numbers
        # that are generated
        Statement("v1", "=", "0*rand()*N", "", float),
        Statement("v1", "=", "0/rand()*N", "", float),
    ]
    scalar, vector = optimise_statements([], statements, variables)
    for vs in vector[:3]:
        assert (
            vs.expr.count("rand()") == 2
        ), f"Expression should still contain two rand() calls, but got {str(vs)}"
    for vs in vector[3:]:
        assert (
            vs.expr.count("rand()") == 1
        ), f"Expression should still contain a rand() call, but got {str(vs)}"


@pytest.mark.codegen_independent
def test_apply_loop_invariant_optimisation_simplification():
    variables = {
        "v1": Variable("v1", scalar=False),
        "v2": Variable("v2", scalar=False),
        "i1": Variable("i1", scalar=False, dtype=int),
        "N": Constant("N", 10),
    }
    statements = [
        # Should be simplified to 0.0
        Statement("v1", "=", "v1 - v1", "", float),
        Statement("v1", "=", "N*v1 - N*v1", "", float),
        Statement("v1", "=", "v1*N * 0", "", float),
        Statement("v1", "=", "v1 * 0", "", float),
        Statement("v1", "=", "v1 * 0.0", "", float),
        Statement("v1", "=", "0.0 / (v1*N)", "", float),
        # Should be simplified to 0
        Statement("i1", "=", "i1*N * 0", "", int),
        Statement("i1", "=", "0 * i1", "", int),
        Statement("i1", "=", "0 * i1*N", "", int),
        Statement("i1", "=", "i1 * 0", "", int),
        # Should be simplified to v1*N
        Statement("v2", "=", "0 + v1*N", "", float),
        Statement("v2", "=", "v1*N + 0.0", "", float),
        Statement("v2", "=", "v1*N - 0", "", float),
        Statement("v2", "=", "v1*N - 0.0", "", float),
        Statement("v2", "=", "1 * v1*N", "", float),
        Statement("v2", "=", "1.0 * v1*N", "", float),
        Statement("v2", "=", "v1*N / 1.0", "", float),
        Statement("v2", "=", "v1*N / 1", "", float),
        # Should be simplified to i1
        Statement("i1", "=", "i1*1", "", int),
        Statement("i1", "=", "i1//1", "", int),
        Statement("i1", "=", "i1+0", "", int),
        Statement("i1", "=", "0+i1", "", int),
        Statement("i1", "=", "i1-0", "", int),
        # Should *not* be simplified (because it would change the type,
        # important for integer division, for example)
        Statement("v1", "=", "i1*1.0", "", float),
        Statement("v1", "=", "1.0*i1", "", float),
        Statement("v1", "=", "i1/1.0", "", float),
        Statement("v1", "=", "i1/1", "", float),
        Statement("v1", "=", "i1+0.0", "", float),
        Statement("v1", "=", "0.0+i1", "", float),
        Statement("v1", "=", "i1-0.0", "", float),
        ## Should *not* be simplified, flooring division by 1 changes the value
        Statement("v1", "=", "v2//1.0", "", float),
        Statement("i1", "=", "i1//1.0", "", float),  # changes type
    ]
    scalar, vector = optimise_statements([], statements, variables)
    assert len(scalar) == 0
    for s in vector[:6]:
        assert s.expr == "0.0"
    for s in vector[6:10]:
        assert s.expr == "0", s.expr  # integer
    for s in vector[10:18]:
        expr = s.expr.replace(" ", "")
        assert expr == "v1*N" or expr == "N*v1"
    for s in vector[18:23]:
        expr = s.expr.replace(" ", "")
        assert expr == "i1"
    for s in vector[23:27]:
        expr = s.expr.replace(" ", "")
        assert expr == "1.0*i1" or expr == "i1*1.0" or expr == "i1/1.0"
    for s in vector[27:30]:
        expr = s.expr.replace(" ", "")
        assert expr == "0.0+i1" or expr == "i1+0.0"
    for s in vector[30:31]:
        expr = s.expr.replace(" ", "")
        assert expr == "v2//1.0" or expr == "v2//1"
    for s in vector[31:]:
        expr = s.expr.replace(" ", "")
        assert expr == "i1//1.0"


@pytest.mark.codegen_independent
def test_apply_loop_invariant_optimisation_constant_evaluation():
    variables = {
        "v1": Variable("v1", scalar=False),
        "v2": Variable("v2", scalar=False),
        "i1": Variable("i1", scalar=False, dtype=int),
        "N": Constant("N", 10),
        "s1": Variable("s1", scalar=True, dtype=float),
        "s2": Variable("s2", scalar=True, dtype=float),
        "exp": DEFAULT_FUNCTIONS["exp"],
    }
    statements = [
        Statement("v1", "=", "v1 * (1 + 2 + 3)", "", float),
        Statement("v1", "=", "exp(N)*v1", "", float),
        Statement("v1", "=", "exp(0)*v1", "", float),
    ]
    scalar, vector = optimise_statements([], statements, variables)
    # exp(N) should be pulled out of the vector statements, the rest should be
    # evaluated in place
    assert len(scalar) == 1
    assert scalar[0].expr == "exp(N)"
    assert len(vector) == 3
    expr = vector[0].expr.replace(" ", "")
    assert expr == "_lio_1*v1" or "v1*_lio_1"
    expr = vector[1].expr.replace(" ", "")
    assert expr == "6.0*v1" or "v1*6.0"
    assert vector[2].expr == "v1"


@pytest.mark.codegen_independent
def test_automatic_augmented_assignments():
    # We test that statements that could be rewritten as augmented assignments
    # are correctly rewritten (using sympy to test for symbolic equality)
    variables = {
        "x": ArrayVariable("x", owner=None, size=10, device=device),
        "y": ArrayVariable("y", owner=None, size=10, device=device),
        "z": ArrayVariable("y", owner=None, size=10, device=device),
        "b": ArrayVariable("b", owner=None, size=10, dtype=bool, device=device),
        "clip": DEFAULT_FUNCTIONS["clip"],
        "inf": DEFAULT_CONSTANTS["inf"],
    }
    statements = [
        # examples that should be rewritten
        # Note that using our approach, we will never get -= or /= but always
        # the equivalent += or *= statements
        ("x = x + 1.0", "x += 1.0"),
        ("x = 2.0 * x", "x *= 2.0"),
        ("x = x - 3.0", "x += -3.0"),
        ("x = x/2.0", "x *= 0.5"),
        ("x = y + (x + 1.0)", "x += y + 1.0"),
        ("x = x + x", "x *= 2"),
        ("x = x + y + z", "x += y + z"),
        ("x = x + y + z", "x += y + z"),
        # examples that should not be rewritten
        ("x = 1.0/x", "x = 1.0/x"),
        ("x = 1.0", "x = 1.0"),
        ("x = 2.0*(x + 1.0)", "x = 2.0*(x + 1.0)"),
        ("x = clip(x + y, 0.0, inf)", "x = clip(x + y, 0.0, inf)"),
        ("b = b or False", "b = b or False"),
    ]
    for orig, rewritten in statements:
        scalar, vector = make_statements(orig, variables, np.float32)
        try:  # we augment the assertion error with the original statement
            assert (
                len(scalar) == 0
            ), f"Did not expect any scalar statements but got {str(scalar)}"
            assert (
                len(vector) == 1
            ), f"Did expect a single statement but got {str(vector)}"
            statement = vector[0]
            expected_var, expected_op, expected_expr, _ = parse_statement(rewritten)
            assert (
                expected_var == statement.var
            ), f"expected write to variable {expected_var}, not to {statement.var}"
            assert (
                expected_op == statement.op
            ), f"expected operation {expected_op}, not {statement.op}"
            # Compare the two expressions using sympy to allow for different order etc.
            sympy_expected = str_to_sympy(expected_expr)
            sympy_actual = str_to_sympy(statement.expr)
            assert sympy_expected == sympy_actual, (
                f"RHS expressions '{sympy_to_str(sympy_expected)}' and"
                f" '{sympy_to_str(sympy_actual)}' are not identical"
            )
        except AssertionError as ex:
            raise AssertionError(
                f"Transformation for statement '{orig}' gave an unexpected result: {ex}"
            )


@pytest.mark.codegen_independent
@pytest.mark.parametrize(
    "s",
    [
        "x, y = 3",
        "x * y",
        "x = ",
        "x.a = 3",
        "x++",
        "x[0] = 3",
        "dx/dt = -v / tau",
        "v == 3*mV",
    ],
)
def test_incorrect_statements(s):
    with pytest.raises(ValueError):
        parse_statement(s)


def test_clear_cache():
    target = prefs.codegen.target
    if target == "numpy":
        assert "numpy" not in _cache_dirs_and_extensions
        with pytest.raises(ValueError):
            clear_cache("numpy")
    else:
        assert target in _cache_dirs_and_extensions
        cache_dir, _ = _cache_dirs_and_extensions[target]
        # Create a file that should not be there
        fname = os.path.join(cache_dir, "some_file.py")
        open(fname, "w").close()
        # clear_cache should refuse to clear the directory
        with pytest.raises(IOError):
            clear_cache(target)

        os.remove(fname)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="CC and CXX variables are ignored on Windows.",
)
def test_compiler_error():
    # In particular on OSX with clang in a conda environment, compilation might fail.
    # Switching to a system gcc might help in such cases. Make sure that the error
    # message mentions that.
    old_CC = os.environ.get("CC", None)
    old_CXX = os.environ.get("CXX", None)
    os.environ.update({"CC": "non-existing-compiler", "CXX": "non-existing-compiler++"})
    try:
        with catch_logs() as l:
            assert not CythonCodeObject.is_available()
        assert len(l) > 0  # There are additional warnings about compiler flags
        last_warning = l[-1]
        assert last_warning[1].endswith(".failed_compile_test")
        assert "CC" in last_warning[2] and "CXX" in last_warning[2]

    finally:
        if old_CC:
            os.environ["CC"] = old_CC
        else:
            del os.environ["CC"]
        if old_CXX:
            os.environ["CXX"] = old_CXX
        else:
            del os.environ["CXX"]


def test_compiler_c99():
    # On a user's computer, we do not know whether the compiler actually
    # has C99 support, so we just check whether the test does not raise an
    # error

    # The compiler check previously created spurious '-.o' files (see #1348)
    if os.path.exists("-.o"):
        os.remove("-.o")
    c99_support = compiler_supports_c99()
    assert not os.path.exists("-.o")
    # On our Azure test server we know that the compilers support C99
    if os.environ.get("AGENT_OS", ""):
        assert c99_support


def test_cpp_flags_support():
    from distutils.ccompiler import get_default_compiler

    from brian2.codegen.cpp_prefs import _compiler_flag_compatibility

    _compiler_flag_compatibility.clear()  # make sure cache is empty
    compiler = get_default_compiler()
    if compiler == "msvc":
        pytest.skip("No flag support check for msvc")
    old_prefs = prefs["codegen.cpp.extra_compile_args"]

    # Should always be supported
    prefs["codegen.cpp.extra_compile_args"] = ["-w"]
    _, compile_args = get_compiler_and_args()
    assert compile_args == prefs["codegen.cpp.extra_compile_args"]

    # Should never be supported and raise a warning
    prefs["codegen.cpp.extra_compile_args"] = ["-invalidxyz"]
    with catch_logs() as l:
        _, compile_args = get_compiler_and_args()
    assert len(l) == 1 and l[0][0] == "WARNING"
    assert compile_args == []

    prefs["codegen.cpp.extra_compile_args"] = old_prefs


@pytest.mark.skipif(
    platform.system() != "Windows", reason="MSVC flags are only relevant on Windows"
)
@pytest.mark.skipif(
    prefs["codegen.target"] == "numpy", reason="Test only relevant for compiled code"
)
def test_msvc_flags():
    # Very basic test that flags are stored to disk
    import brian2.codegen.cpp_prefs as cpp_prefs

    user_dir = os.path.join(os.path.expanduser("~"), ".brian")
    flag_file = os.path.join(user_dir, "cpu_flags.txt")
    assert len(cpp_prefs.msvc_arch_flag)
    assert os.path.exists(flag_file)
    with open(flag_file, encoding="utf-8") as f:
        previously_stored_flags = json.load(f)
    hostname = socket.gethostname()
    assert hostname in previously_stored_flags
    assert len(previously_stored_flags[hostname])


@pytest.mark.codegen_independent
@pytest.mark.parametrize(
    "renderer",
    [
        NodeRenderer(),
        NumpyNodeRenderer(),
        CythonNodeRenderer(),
        CPPNodeRenderer(),
    ],
)
def test_number_rendering(renderer):
    import ast

    for number in [0.5, np.float32(0.5), np.float64(0.5)]:
        # In numpy 2.0, repr(np.float64(0.5)) is 'np.float64(0.5)'
        node = ast.Constant(value=number)
        assert renderer.render_node(node) == "0.5"


if __name__ == "__main__":
    test_auto_target()
    test_analyse_identifiers()
    test_get_identifiers_recursively()
    test_write_to_subexpression()
    test_repeated_subexpressions()
    test_nested_subexpressions()
    test_apply_loop_invariant_optimisation()
    test_apply_loop_invariant_optimisation_integer()
    test_apply_loop_invariant_optimisation_boolean()
    test_apply_loop_invariant_optimisation_no_optimisation()
    test_apply_loop_invariant_optimisation_simplification()
    test_apply_loop_invariant_optimisation_constant_evaluation()
    test_automatic_augmented_assignments()
    test_clear_cache()
    test_msvc_flags()
