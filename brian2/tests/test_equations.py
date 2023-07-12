import sys

import numpy as np

try:
    from IPython.lib.pretty import pprint
except ImportError:
    pprint = None
import pytest

from brian2 import Equations, Expression, Hz, Unit, farad, metre, ms, mV, second, volt
from brian2.core.namespace import DEFAULT_UNITS
from brian2.equations.equations import (
    BOOLEAN,
    DIFFERENTIAL_EQUATION,
    FLOAT,
    INTEGER,
    PARAMETER,
    SUBEXPRESSION,
    EquationError,
    SingleEquation,
    check_identifier_basic,
    check_identifier_constants,
    check_identifier_functions,
    check_identifier_reserved,
    check_identifier_units,
    dimensions_and_type_from_string,
    extract_constant_subexpressions,
    parse_string_equations,
)
from brian2.equations.refractory import check_identifier_refractory
from brian2.groups.group import Group
from brian2.units.fundamentalunits import (
    DIMENSIONLESS,
    DimensionMismatchError,
    get_dimensions,
)


# a simple Group for testing
class SimpleGroup(Group):
    def __init__(self, variables, namespace=None):
        self.variables = variables
        self.namespace = namespace


@pytest.mark.codegen_independent
def test_utility_functions():
    unit_namespace = DEFAULT_UNITS

    # Some simple tests whether the namespace returned by
    # get_default_namespace() makes sense
    assert "volt" in unit_namespace
    assert "ms" in unit_namespace
    assert unit_namespace["ms"] is ms
    assert unit_namespace["ms"] is unit_namespace["msecond"]
    for unit in unit_namespace.values():
        assert isinstance(unit, Unit)

    assert dimensions_and_type_from_string("second") == (second.dim, FLOAT)
    assert dimensions_and_type_from_string("1") == (DIMENSIONLESS, FLOAT)
    assert dimensions_and_type_from_string("volt") == (volt.dim, FLOAT)
    assert dimensions_and_type_from_string("second ** -1") == (Hz.dim, FLOAT)
    assert dimensions_and_type_from_string("farad / metre**2") == (
        (farad / metre**2).dim,
        FLOAT,
    )
    assert dimensions_and_type_from_string("boolean") == (DIMENSIONLESS, BOOLEAN)
    assert dimensions_and_type_from_string("integer") == (DIMENSIONLESS, INTEGER)
    with pytest.raises(ValueError):
        dimensions_and_type_from_string("metr / second")
    with pytest.raises(ValueError):
        dimensions_and_type_from_string("metre **")
    with pytest.raises(ValueError):
        dimensions_and_type_from_string("5")
    with pytest.raises(ValueError):
        dimensions_and_type_from_string("2 / second")
    # Only the use of base units is allowed
    with pytest.raises(ValueError):
        dimensions_and_type_from_string("farad / cm**2")


@pytest.mark.codegen_independent
def test_identifier_checks():
    legal_identifiers = ["v", "Vm", "V", "x", "ge", "g_i", "a2", "gaba_123"]
    illegal_identifiers = ["_v", "1v", "ü", "ge!", "v.x", "for", "else", "if"]

    for identifier in legal_identifiers:
        try:
            check_identifier_basic(identifier)
            check_identifier_reserved(identifier)
        except ValueError as ex:
            raise AssertionError(
                f'check complained about identifier "{identifier}": {ex}'
            )

    for identifier in illegal_identifiers:
        with pytest.raises(SyntaxError):
            check_identifier_basic(identifier)

    for identifier in ("t", "dt", "xi", "i", "N"):
        with pytest.raises(SyntaxError):
            check_identifier_reserved(identifier)

    for identifier in ("not_refractory", "refractory", "refractory_until"):
        with pytest.raises(SyntaxError):
            check_identifier_refractory(identifier)

    for identifier in ("exp", "sin", "sqrt"):
        with pytest.raises(SyntaxError):
            check_identifier_functions(identifier)

    for identifier in ("e", "pi", "inf"):
        with pytest.raises(SyntaxError):
            check_identifier_constants(identifier)

    for identifier in ("volt", "second", "mV", "nA"):
        with pytest.raises(SyntaxError):
            check_identifier_units(identifier)

    # Check identifier registry
    assert check_identifier_basic in Equations.identifier_checks
    assert check_identifier_reserved in Equations.identifier_checks
    assert check_identifier_refractory in Equations.identifier_checks
    assert check_identifier_functions in Equations.identifier_checks
    assert check_identifier_constants in Equations.identifier_checks
    assert check_identifier_units in Equations.identifier_checks

    # Set up a dummy identifier check that disallows the variable name
    # gaba_123 (that is otherwise valid)
    def disallow_gaba_123(identifier):
        if identifier == "gaba_123":
            raise SyntaxError("I do not like this name")

    Equations.check_identifier("gaba_123")
    old_checks = set(Equations.identifier_checks)
    Equations.register_identifier_check(disallow_gaba_123)
    with pytest.raises(SyntaxError):
        Equations.check_identifier("gaba_123")
    Equations.identifier_checks = old_checks

    # registering a non-function should not work
    with pytest.raises(ValueError):
        Equations.register_identifier_check("no function")


@pytest.mark.codegen_independent
def test_parse_equations():
    """Test the parsing of equation strings"""
    # A simple equation
    eqs = parse_string_equations("dv/dt = -v / tau : 1")
    assert len(eqs) == 1 and "v" in eqs and eqs["v"].type == DIFFERENTIAL_EQUATION
    assert eqs["v"].dim is DIMENSIONLESS

    # A complex one
    eqs = parse_string_equations(
        """
        dv/dt = -(v +
                  ge + # excitatory conductance
                  I # external current
                 )/ tau : volt
        dge/dt = -ge / tau_ge : volt
        I = sin(2 * pi * f * t) : volt
        f : Hz (constant)
        b : boolean
        n : integer
        """
    )
    assert len(eqs) == 6
    assert "v" in eqs and eqs["v"].type == DIFFERENTIAL_EQUATION
    assert "ge" in eqs and eqs["ge"].type == DIFFERENTIAL_EQUATION
    assert "I" in eqs and eqs["I"].type == SUBEXPRESSION
    assert "f" in eqs and eqs["f"].type == PARAMETER
    assert "b" in eqs and eqs["b"].type == PARAMETER
    assert "n" in eqs and eqs["n"].type == PARAMETER
    assert eqs["f"].var_type == FLOAT
    assert eqs["b"].var_type == BOOLEAN
    assert eqs["n"].var_type == INTEGER
    assert eqs["v"].dim is volt.dim
    assert eqs["ge"].dim is volt.dim
    assert eqs["I"].dim is volt.dim
    assert eqs["f"].dim is Hz.dim
    assert eqs["v"].flags == []
    assert eqs["ge"].flags == []
    assert eqs["I"].flags == []
    assert eqs["f"].flags == ["constant"]

    duplicate_eqs = """
    dv/dt = -v / tau : 1
    v = 2 * t : 1
    """
    with pytest.raises(EquationError):
        parse_string_equations(duplicate_eqs)
    parse_error_eqs = [
        """
        dv/d = -v / tau : 1
        x = 2 * t : 1
        """,
        """
        dv/dt = -v / tau : 1 : volt
        x = 2 * t : 1
        """,
        "dv/dt = -v / tau : 2 * volt",
        "dv/dt = v / second : boolean",
    ]
    for error_eqs in parse_error_eqs:
        with pytest.raises((ValueError, EquationError, TypeError)):
            parse_string_equations(error_eqs)


@pytest.mark.codegen_independent
def test_correct_replacements():
    """Test replacing variables via keyword arguments"""
    # replace a variable name with a new name
    eqs = Equations("dv/dt = -v / tau : 1", v="V")
    # Correct left hand side
    assert ("V" in eqs) and not ("v" in eqs)
    # Correct right hand side
    assert ("V" in eqs["V"].identifiers) and not ("v" in eqs["V"].identifiers)

    # replace a variable name with a value
    eqs = Equations("dv/dt = -v / tau : 1", tau=10 * ms)
    assert not "tau" in eqs["v"].identifiers


@pytest.mark.codegen_independent
def test_wrong_replacements():
    """Tests for replacements that should not work"""
    # Replacing a variable name with an illegal new name
    with pytest.raises(SyntaxError):
        Equations("dv/dt = -v / tau : 1", v="illegal name")
    with pytest.raises(SyntaxError):
        Equations("dv/dt = -v / tau : 1", v="_reserved")
    with pytest.raises(SyntaxError):
        Equations("dv/dt = -v / tau : 1", v="t")

    # Replacing a variable name with a value that already exists
    with pytest.raises(EquationError):
        Equations(
            """
            dv/dt = -v / tau : 1
            dx/dt = -x / tau : 1
            """,
            v="x",
        )

    # Replacing a model variable name with a value
    with pytest.raises(ValueError):
        Equations("dv/dt = -v / tau : 1", v=3 * mV)

    # Replacing with an illegal value
    with pytest.raises(SyntaxError):
        Equations("dv/dt = -v/tau : 1", tau=np.arange(5))


@pytest.mark.codegen_independent
def test_substitute():
    # Check that Equations.substitute returns an independent copy
    eqs = Equations("dx/dt = x : 1")
    eqs2 = eqs.substitute(x="y")

    # First equation should be unaffected
    assert len(eqs) == 1 and "x" in eqs
    assert eqs["x"].expr == Expression("x")

    # Second equation should have x substituted by y
    assert len(eqs2) == 1 and "y" in eqs2
    assert eqs2["y"].expr == Expression("y")


@pytest.mark.codegen_independent
def test_construction_errors():
    """
    Test that the Equations constructor raises errors correctly
    """
    # parse error
    with pytest.raises(EquationError):
        Equations("dv/dt = -v / tau volt")
    with pytest.raises(EquationError):
        Equations("dv/dt = -v / tau : volt second")

    # incorrect unit definition
    with pytest.raises(EquationError):
        Equations("dv/dt = -v / tau : mvolt")
    with pytest.raises(EquationError):
        Equations("dv/dt = -v / tau : voltage")
    with pytest.raises(EquationError):
        Equations("dv/dt = -v / tau : 1.0*volt")

    # Only a single string or a list of SingleEquation objects is allowed
    with pytest.raises(TypeError):
        Equations(None)
    with pytest.raises(TypeError):
        Equations(42)
    with pytest.raises(TypeError):
        Equations(["dv/dt = -v / tau : volt"])

    # duplicate variable names
    with pytest.raises(EquationError):
        Equations(
            """
            dv/dt = -v / tau : volt
            v = 2 * t/second * volt : volt
            """
        )

    eqs = [
        SingleEquation(
            DIFFERENTIAL_EQUATION, "v", volt.dim, expr=Expression("-v / tau")
        ),
        SingleEquation(
            SUBEXPRESSION, "v", volt.dim, expr=Expression("2 * t/second * volt")
        ),
    ]
    with pytest.raises(EquationError):
        Equations(eqs)

    # illegal variable names
    with pytest.raises(SyntaxError):
        Equations("ddt/dt = -dt / tau : volt")
    with pytest.raises(SyntaxError):
        Equations("dt/dt = -t / tau : volt")
    with pytest.raises(SyntaxError):
        Equations("dxi/dt = -xi / tau : volt")
    with pytest.raises(SyntaxError):
        Equations("for : volt")
    with pytest.raises((EquationError, SyntaxError)):
        Equations("d1a/dt = -1a / tau : volt")
    with pytest.raises(SyntaxError):
        Equations("d_x/dt = -_x / tau : volt")

    # xi in a subexpression
    with pytest.raises(EquationError):
        Equations(
            """
            dv/dt = -(v + I) / (5 * ms) : volt
            I = second**-1*xi**-2*volt : volt
            """
        )

    # more than one xi
    with pytest.raises(EquationError):
        Equations(
            """
            dv/dt = -v / tau + xi/tau**.5 : volt
            dx/dt = -x / tau + 2*xi/tau : volt
            tau : second
            """
        )
    # using not-allowed flags
    eqs = Equations("dv/dt = -v / (5 * ms) : volt (flag)")
    eqs.check_flags({DIFFERENTIAL_EQUATION: ["flag"]})  # allow this flag
    with pytest.raises(ValueError):
        eqs.check_flags({DIFFERENTIAL_EQUATION: []})
    with pytest.raises(ValueError):
        eqs.check_flags({})
    with pytest.raises(ValueError):
        eqs.check_flags({SUBEXPRESSION: ["flag"]})
    with pytest.raises(ValueError):
        eqs.check_flags({DIFFERENTIAL_EQUATION: ["otherflag"]})
    eqs = Equations("dv/dt = -v / (5 * ms) : volt (flag1, flag2)")
    eqs.check_flags({DIFFERENTIAL_EQUATION: ["flag1", "flag2"]})  # allow both flags
    # Don't allow the two flags in combination
    with pytest.raises(ValueError):
        eqs.check_flags(
            {DIFFERENTIAL_EQUATION: ["flag1", "flag2"]},
            incompatible_flags=[("flag1", "flag2")],
        )
    eqs = Equations(
        """
        dv/dt = -v / (5 * ms) : volt (flag1)
        dw/dt = -w / (5 * ms) : volt (flag2)
        """
    )
    # They should be allowed when used independently
    eqs.check_flags(
        {DIFFERENTIAL_EQUATION: ["flag1", "flag2"]},
        incompatible_flags=[("flag1", "flag2")],
    )

    # Circular subexpression
    with pytest.raises(ValueError):
        Equations(
            """
            dv/dt = -(v + w) / (10 * ms) : 1
            w = 2 * x : 1
            x = 3 * w : 1
            """
        )

    # Boolean/integer differential equations
    with pytest.raises(TypeError):
        Equations("dv/dt = -v / (10*ms) : boolean")
    with pytest.raises(TypeError):
        Equations("dv/dt = -v / (10*ms) : integer")


@pytest.mark.codegen_independent
def test_unit_checking():
    # dummy Variable class
    class S:
        def __init__(self, dimensions):
            self.dim = get_dimensions(dimensions)

    # inconsistent unit for a differential equation
    eqs = Equations("dv/dt = -v : volt")
    group = SimpleGroup({"v": S(volt)})
    with pytest.raises(DimensionMismatchError):
        eqs.check_units(group, {})

    eqs = Equations("dv/dt = -v / tau: volt")
    group = SimpleGroup(namespace={"tau": 5 * mV}, variables={"v": S(volt)})
    with pytest.raises(DimensionMismatchError):
        eqs.check_units(group, {})
    group = SimpleGroup(namespace={"I": 3 * second}, variables={"v": S(volt)})
    eqs = Equations("dv/dt = -(v + I) / (5 * ms): volt")
    with pytest.raises(DimensionMismatchError):
        eqs.check_units(group, {})

    eqs = Equations(
        """
        dv/dt = -(v + I) / (5 * ms): volt
        I : second
        """
    )
    group = SimpleGroup(variables={"v": S(volt), "I": S(second)}, namespace={})
    with pytest.raises(DimensionMismatchError):
        eqs.check_units(group, {})

    # inconsistent unit for a subexpression
    eqs = Equations(
        """
        dv/dt = -v / (5 * ms) : volt
        I = 2 * v : amp
        """
    )
    group = SimpleGroup(variables={"v": S(volt), "I": S(second)}, namespace={})
    with pytest.raises(DimensionMismatchError):
        eqs.check_units(group, {})


@pytest.mark.codegen_independent
def test_properties():
    """
    Test accessing the various properties of equation objects
    """
    tau = 10 * ms
    eqs = Equations(
        """
        dv/dt = -(v + I)/ tau : volt
        I = sin(2 * 22/7. * f * t)* volt : volt
        f = freq * Hz: Hz
        freq : 1
        """
    )
    assert (
        len(eqs.diff_eq_expressions) == 1
        and eqs.diff_eq_expressions[0][0] == "v"
        and isinstance(eqs.diff_eq_expressions[0][1], Expression)
    )
    assert eqs.diff_eq_names == {"v"}
    assert (
        len(eqs.eq_expressions) == 3
        and {name for name, _ in eqs.eq_expressions} == {"v", "I", "f"}
        and all((isinstance(expr, Expression) for _, expr in eqs.eq_expressions))
    )
    assert len(eqs.eq_names) == 3 and eqs.eq_names == {"v", "I", "f"}
    assert set(eqs.keys()) == {"v", "I", "f", "freq"}
    # test that the equations object is iterable itself
    assert all(isinstance(eq, SingleEquation) for eq in eqs.values())
    assert all(isinstance(eq, str) for eq in eqs)
    assert (
        len(eqs.ordered) == 4
        and all(isinstance(eq, SingleEquation) for eq in eqs.ordered)
        and [eq.varname for eq in eqs.ordered] == ["f", "I", "v", "freq"]
    )
    assert [eq.unit for eq in eqs.ordered] == [Hz, volt, volt, 1]
    assert eqs.names == {"v", "I", "f", "freq"}
    assert eqs.parameter_names == {"freq"}
    assert eqs.subexpr_names == {"I", "f"}
    dimensions = eqs.dimensions
    assert set(dimensions.keys()) == {"v", "I", "f", "freq"}
    assert dimensions["v"] is volt.dim
    assert dimensions["I"] is volt.dim
    assert dimensions["f"] is Hz.dim
    assert dimensions["freq"] is DIMENSIONLESS
    assert eqs.names == set(eqs.dimensions.keys())
    assert eqs.identifiers == {"tau", "volt", "Hz", "sin", "t"}

    # stochastic equations
    assert len(eqs.stochastic_variables) == 0
    assert eqs.stochastic_type is None

    eqs = Equations("""dv/dt = -v / tau + 0.1*second**-.5*xi : 1""")
    assert eqs.stochastic_variables == {"xi"}
    assert eqs.stochastic_type == "additive"

    eqs = Equations(
        "dv/dt = -v / tau + 0.1*second**-.5*xi_1 +  0.1*second**-.5*xi_2: 1"
    )
    assert eqs.stochastic_variables == {"xi_1", "xi_2"}
    assert eqs.stochastic_type == "additive"

    eqs = Equations("dv/dt = -v / tau + 0.1*second**-1.5*xi*t : 1")
    assert eqs.stochastic_type == "multiplicative"

    eqs = Equations("dv/dt = -v / tau + 0.1*second**-1.5*xi*v : 1")
    assert eqs.stochastic_type == "multiplicative"


@pytest.mark.codegen_independent
def test_concatenation():
    eqs1 = Equations(
        """
        dv/dt = -(v + I) / tau : volt
        I = sin(2*pi*freq*t) : volt
        freq : Hz
        """
    )

    # Concatenate two equation objects
    eqs2 = Equations("dv/dt = -(v + I) / tau : volt") + Equations(
        """
        I = sin(2*pi*freq*t) : volt
        freq : Hz
        """
    )

    # Concatenate using "in-place" addition (which is not actually in-place)
    eqs3 = Equations("dv/dt = -(v + I) / tau : volt")
    eqs3 += Equations(
        """
        I = sin(2*pi*freq*t) : volt
        freq : Hz
        """
    )

    # Concatenate with a string (will be parsed first)
    eqs4 = Equations("dv/dt = -(v + I) / tau : volt")
    eqs4 += """I = sin(2*pi*freq*t) : volt
               freq : Hz"""

    # Concatenating with something that is not a string should not work
    with pytest.raises(TypeError):
        eqs4 + 5

    # The string representation is canonical, therefore it should be identical
    # in all cases
    assert str(eqs1) == str(eqs2)
    assert str(eqs2) == str(eqs3)
    assert str(eqs3) == str(eqs4)


@pytest.mark.codegen_independent
def test_extract_subexpressions():
    eqs = Equations(
        """
        dv/dt = -v / (10*ms) : 1
        s1 = 2*v : 1
        s2 = -v : 1 (constant over dt)
        """
    )
    variable, constant = extract_constant_subexpressions(eqs)
    assert [var in variable for var in ["v", "s1", "s2"]]
    assert variable["s1"].type == SUBEXPRESSION
    assert variable["s2"].type == PARAMETER
    assert constant["s2"].type == SUBEXPRESSION


@pytest.mark.codegen_independent
def test_repeated_construction():
    eqs1 = Equations("dx/dt = x : 1")
    eqs2 = Equations("dx/dt = x : 1", x="y")
    assert len(eqs1) == 1
    assert "x" in eqs1
    assert eqs1["x"].expr == Expression("x")
    assert len(eqs2) == 1
    assert "y" in eqs2
    assert eqs2["y"].expr == Expression("y")


@pytest.mark.codegen_independent
def test_str_repr():
    """
    Test the string representation (only that it does not throw errors).
    """
    tau = 10 * ms
    eqs = Equations(
        """
        dv/dt = -(v + I)/ tau : volt (unless refractory)
        I = sin(2 * 22/7. * f * t)* volt : volt
        f : Hz
        """
    )
    assert len(str(eqs)) > 0
    assert len(repr(eqs)) > 0

    # Test str and repr of SingleEquations explicitly (might already have been
    # called by Equations
    for eq in eqs.values():
        assert (len(str(eq))) > 0
        assert (len(repr(eq))) > 0


@pytest.mark.codegen_independent
def test_dependency_calculation():
    eqs = Equations(
        """
        dv/dt = I_m / C_m : volt
        I_m = I_ext + I_pas : amp
        I_ext = 1*nA + sin(2*pi*100*Hz*t)*nA : amp
        I_pas = g_L*(E_L - v) : amp
        """
    )
    deps = eqs.dependencies
    assert set(deps.keys()) == {"v", "I_m", "I_ext", "I_pas"}

    # v depends directly on I_m, on I_ext and I_pas via I_m, and on v via I_m -> I_pas
    assert len(deps["v"]) == 4
    assert {d.equation.varname for d in deps["v"]} == {"I_m", "I_ext", "I_pas", "v"}
    expected_via = {
        "I_m": (),
        "I_pas": ("I_m",),
        "I_ext": ("I_m",),
        "v": ("I_m", "I_pas"),
    }
    assert all([d.via == expected_via[d.equation.varname] for d in deps["v"]])

    # I_m depends directly on I_ext and I_pas, and on v via I_pas
    assert len(deps["I_m"]) == 3
    assert {d.equation.varname for d in deps["I_m"]} == {"I_ext", "I_pas", "v"}
    expected_via = {"I_ext": (), "I_pas": (), "v": ("I_pas",)}
    assert all([d.via == expected_via[d.equation.varname] for d in deps["I_m"]])

    # I_ext does not depend on anything
    assert len(deps["I_ext"]) == 0

    # I_pas depends on v directly
    assert len(deps["I_pas"]) == 1
    assert deps["I_pas"][0].equation.varname == "v"
    assert deps["I_pas"][0].via == ()


@pytest.mark.codegen_independent
@pytest.mark.skipif(pprint is None, reason="ipython is not installed")
def test_ipython_pprint():
    from io import StringIO

    eqs = Equations(
        """
        dv/dt = -(v + I)/ tau : volt (unless refractory)
        I = sin(2 * 22/7. * f * t)* volt : volt
        f : Hz
        """
    )
    # Test ipython's pretty printing
    old_stdout = sys.stdout
    string_output = StringIO()
    sys.stdout = string_output
    pprint(eqs)
    assert len(string_output.getvalue()) > 0
    sys.stdout = old_stdout


if __name__ == "__main__":
    test_utility_functions()
    test_identifier_checks()
    test_parse_equations()
    test_correct_replacements()
    test_substitute()
    test_wrong_replacements()
    test_construction_errors()
    test_concatenation()
    test_unit_checking()
    test_properties()
    test_extract_subexpressions()
    test_repeated_construction()
    test_str_repr()
