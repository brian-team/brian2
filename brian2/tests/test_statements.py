"""
Tests for the Statements class and its substitution functionality.
"""

import pytest

from brian2.equations.codestrings import Statements
from brian2.units import mV, nS


# Core functionality tests - separate for clarity
def test_statements_basic():
    """Test basic Statements creation without substitution"""
    stmt = Statements("g += w")
    assert str(stmt) == "g += w"


def test_statements_value_substitution():
    """Test substituting an identifier with a numeric value"""
    stmt = Statements("g += k*w", k=0.3)
    assert str(stmt) == "g += (0.3)*w"


def test_statements_name_substitution():
    """Test substituting an identifier with another name"""
    stmt = Statements("g += k*w", g="g_ampa")
    assert str(stmt) == "g_ampa += k*w"


def test_statements_multiple_substitutions():
    """Test multiple substitutions simultaneously"""
    stmt = Statements("g += k*w", g="g_ampa", k=0.3)
    assert str(stmt) == "g_ampa += (0.3)*w"


def test_statements_with_units():
    """Test substitution with Brian2 units"""
    stmt = Statements("v += dv", dv=1 * mV)
    result = str(stmt)
    # Brian2 units are represented with their unit name
    assert "mvolt" in result or "mV" in result


def test_statements_with_semicolons():
    """Test statements separated by semicolons"""
    stmt = Statements("x += 1; y += 2", x="x_new")
    result = str(stmt)
    assert "x_new +=" in result
    assert "y +=" in result


def test_statements_complex_expression():
    """Test substitution in complex mathematical expressions"""
    stmt = Statements("v += dt*(-v/tau + I)", tau=10, I=5)
    result = str(stmt)
    assert "(10)" in result
    assert "(5)" in result
    assert "v" in result
    assert "dt" in result


def test_statements_identifiers_after_substitution():
    """Test identifiers are updated after substitution"""
    stmt = Statements("g += k*w", k=0.3)
    identifiers = stmt.identifiers
    assert "g" in identifiers
    assert "w" in identifiers
    # k should not be in identifiers after substitution
    assert "k" not in identifiers


def test_statements_repr():
    """Test the repr output"""
    stmt = Statements("g += w")
    assert repr(stmt) == "Statements('g += w')"


# Parametrized tests for variations
@pytest.mark.parametrize(
    "value,expected_substring",
    [
        (0.3, "(0.3)"),
        (5, "(5)"),
        (-70, "(-70)"),
        (0.123, "(0.123)"),
    ],
)
def test_statements_numeric_value_types(value, expected_substring):
    """Test different numeric value types in substitution"""
    stmt = Statements("x += k", k=value)
    assert expected_substring in str(stmt)


@pytest.mark.parametrize(
    "code,substitutions,expected",
    [
        # Word boundary tests (numeric values get parentheses)
        ("tau_syn += tau", {"tau": 10}, "tau_syn += (10)"),
        # String with underscores (string replacement for name, value for number)
        ("g_ampa += w_exc", {"g_ampa": "g_total", "w_exc": 0.5}, "g_total += (0.5)"),
        # Chained operators (name substitution)
        ("x += y; y += z", {"x": "a", "y": "b"}, "a += b; b += z"),
        # String values are treated as identifiers (no parentheses)
        ("x += val", {"val": "10*ms"}, "x += 10*ms"),
    ],
)
def test_statements_edge_cases(code, substitutions, expected):
    """Test edge cases like word boundaries and special patterns"""
    stmt = Statements(code, **substitutions)
    assert str(stmt) == expected


@pytest.mark.parametrize(
    "stmt1_code,stmt2_code,should_be_equal",
    [
        ("g += w", "g += w", True),
        ("g += w", "g += k*w", False),
        ("x = 1", "x = 1", True),
    ],
)
def test_statements_equality(stmt1_code, stmt2_code, should_be_equal):
    """Test equality comparison between Statements objects"""
    stmt1 = Statements(stmt1_code)
    stmt2 = Statements(stmt2_code)
    if should_be_equal:
        assert stmt1 == stmt2
        assert hash(stmt1) == hash(stmt2)
    else:
        assert stmt1 != stmt2


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
    stmt = Statements("x += weight # Use a small weight", weight=1 * nS)
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
        test_statements_basic()
        test_statements_value_substitution()
        test_statements_name_substitution()
        test_statements_multiple_substitutions()
        test_statements_with_units()
        test_statements_with_semicolons()
        test_statements_complex_expression()
        test_statements_identifiers_after_substitution()
        test_statements_repr()
        # test_statements_numeric_value_types()
        # test_statements_edge_cases()
        # test_statements_equality()
        test_statements_substitution_lhs_error()
        test_statements_substitution_comments()
        test_statements_substitution_multiple_lines()
