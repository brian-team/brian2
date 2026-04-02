import pytest
import sympy

import brian2
from brian2 import (
    Expression,
    Statements,
    mV,
)


def sympy_equals(expr1, expr2):
    """
    Test that whether two string expressions are equal using sympy, allowing
    e.g. for ``sympy_equals("x * x", "x ** 2") == True``.
    """
    s_expr1 = sympy.nsimplify(sympy.sympify(expr1).expand())
    s_expr2 = sympy.nsimplify(sympy.sympify(expr2).expand())
    return s_expr1 == s_expr2


@pytest.mark.codegen_independent
def test_expr_creation():
    """
    Test creating expressions.
    """
    expr = Expression("v > 5 * mV")
    assert expr.code == "v > 5 * mV"
    assert (
        "v" in expr.identifiers
        and "mV" in expr.identifiers
        and "V" not in expr.identifiers
    )
    with pytest.raises(SyntaxError):
        Expression("v 5 * mV")


@pytest.mark.codegen_independent
def test_split_stochastic():
    expr = Expression("(-v + I) / tau")
    # No stochastic part
    assert expr.split_stochastic() == (expr, None)

    # No non-stochastic part -- note that it should return 0 and not None
    expr = Expression("sigma*xi/tau**.5")
    non_stochastic, stochastic = expr.split_stochastic()
    assert sympy_equals(non_stochastic.code, 0)
    assert "xi" in stochastic
    assert len(stochastic) == 1
    assert sympy_equals(stochastic["xi"].code, "sigma/tau**.5")

    expr = Expression("(-v + I) / tau + sigma*xi/tau**.5")
    non_stochastic, stochastic = expr.split_stochastic()
    assert "xi" in stochastic
    assert len(stochastic) == 1
    assert sympy_equals(non_stochastic.code, "(-v + I) / tau")
    assert sympy_equals(stochastic["xi"].code, "sigma/tau**.5")

    expr = Expression("(-v + I) / tau + sigma*xi_1/tau**.5 + xi_2*sigma2/sqrt(tau_2)")
    non_stochastic, stochastic = expr.split_stochastic()
    assert set(stochastic.keys()) == {"xi_1", "xi_2"}
    assert sympy_equals(non_stochastic.code, "(-v + I) / tau")
    assert sympy_equals(stochastic["xi_1"].code, "sigma/tau**.5")
    assert sympy_equals(stochastic["xi_2"].code, "sigma2/tau_2**.5")

    expr = Expression("-v / tau + 1 / xi")
    with pytest.raises(ValueError):
        expr.split_stochastic()


@pytest.mark.codegen_independent
def test_str_repr():
    """
    Test the string representation of expressions and statements. Assumes that
    __str__ returns the complete expression/statement string and __repr__ a
    string of the form "Expression(...)" or "Statements(...)" that can be
    evaluated.
    """
    expr_string = "(v - I)/ tau"
    expr = Expression(expr_string)

    # use sympy to check for equivalence of expressions (terms may have be
    # re-arranged by sympy)
    assert sympy_equals(expr_string, str(expr))
    assert sympy_equals(expr_string, eval(repr(expr)).code)

    # Use exact string equivalence for statements
    statement_string = "v += w"
    statement = Statements(statement_string)

    assert str(statement) == "v += w"
    assert repr(statement) == "Statements('v += w')"


@pytest.mark.codegen_independent
def test_statements_substitution():
    """
    Test that Statements correctly handles substitutions.
    """
    # Test string substitution (rename variable)
    stmt = Statements("v += w", v="x")
    assert str(stmt) == "x += w"

    # Test value substitution
    stmt = Statements("v += k*w", k=0.3)
    assert "0.3" in str(stmt)

    # Test both types of substitutions
    stmt = Statements("v += k*w", v="x", k=0.3)
    assert "x" in str(stmt)
    assert "0.3" in str(stmt)


@pytest.mark.codegen_independent
def test_statements_substitution_lhs_error():
    """
    Test that Statements raises an error when trying to substitute a value
    for a variable on the left-hand side of an assignment.
    """
    # Trying to replace LHS variable with a value should raise an error
    with pytest.raises(ValueError, match="Cannot substitute value"):
        Statements("v += x", v=3 * mV)

    with pytest.raises(ValueError, match="Cannot substitute value"):
        Statements("v = x", v=5)

    # This should work fine (string substitution on LHS)
    stmt = Statements("v += x", v="y")
    assert str(stmt) == "y += x"

    # This should work fine (value substitution on RHS)
    stmt = Statements("v += x", x=3 * mV)
    assert "(3. * mvolt)" in str(stmt)


@pytest.mark.codegen_independent
def test_statements_substitution_comments():
    """
    Test that value substitutions do not affect comments, but name
    substitutions do.
    """
    # Value substitution should not affect comments
    stmt = Statements("x += weight # Use a small weight", weight=1 * brian2.nS)
    code = str(stmt)
    # Comment should remain unchanged
    assert "# Use a small weight" in code
    # Code should have the substitution
    assert "(1. * nsiemens)" in code

    # Name substitution should affect both code and comments
    stmt = Statements("x += weight # x is the post-synaptic target variable", x="y")
    assert str(stmt) == "y += weight # y is the post-synaptic target variable"

    # Multiple lines with comments
    stmt = Statements(
        """
        x += weight
        y += x  # x is the variable
        """,
        x="z",
        weight=0.5,
    )
    code = str(stmt)
    assert "z" in code
    assert "0.5" in code
    assert "# z is the variable" in code


@pytest.mark.codegen_independent
def test_statements_substitution_multiple_lines():
    """
    Test substitutions in multi-line statements.
    """
    stmt = Statements(
        """
        v += w
        u += v
        """,
        v="x",
    )
    code = str(stmt)
    # Both occurrences of v should be replaced
    assert "x += w" in code
    assert "u += x" in code


if __name__ == "__main__":
    test_expr_creation()
    test_split_stochastic()
    test_str_repr()
    test_statements_substitution()
    test_statements_substitution_lhs_error()
    test_statements_substitution_comments()
    test_statements_substitution_multiple_lines()
