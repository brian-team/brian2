import pytest

import numpy as np
from numpy.testing import assert_equal
import sympy

from brian2 import Expression, Statements
from brian2 import Hz, ms, mV, volt, second, get_dimensions, DimensionMismatchError

from brian2.utils.logger import catch_logs
from brian2.core.preferences import prefs

import brian2


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
        and not "V" in expr.identifiers
    )
    with pytest.raises(SyntaxError):
        Expression("v 5 * mV")


@pytest.mark.codegen_independent
def test_split_stochastic():
    tau = 5 * ms
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


if __name__ == "__main__":
    test_expr_creation()
    test_split_stochastic()
    test_str_repr()
