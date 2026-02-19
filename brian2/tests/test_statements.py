"""
Tests for the Statements class and its substitution functionality.
"""

import pytest

from brian2.equations.codestrings import Statements
from brian2.units import mV, ms, nS


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


def test_statements_multiline():
    """Test statements with multiple lines"""
    stmt = Statements(
        """
        g_ampa += w1
        g_nmda += w2
    """,
        w1="0.5*nS",
        w2="0.3*nS",
    )
    result = str(stmt)
    assert "g_ampa +=" in result
    assert "g_nmda +=" in result
    assert "0.5*nS" in result
    assert "0.3*nS" in result


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
        ("tau = tau + tau_syn", {"tau": 5}, "(5) = (5) + tau_syn"),
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
