"""
Module implementing Brian's refractory mechanism.
"""

from brian2.units.allunits import second
from brian2.units.fundamentalunits import DIMENSIONLESS

from .equations import (
    BOOLEAN,
    DIFFERENTIAL_EQUATION,
    PARAMETER,
    Equations,
    Expression,
    SingleEquation,
)

__all__ = ["add_refractoriness"]


def check_identifier_refractory(identifier):
    """
    Check that the identifier is not using a name reserved for the refractory
    mechanism. The reserved names are `not_refractory`, `refractory`,
    `refractory_until`.

    Parameters
    ----------
    identifier : str
        The identifier to check.

    Raises
    ------
    ValueError
        If the identifier is a variable name used for the refractory mechanism.
    """
    if identifier in ("not_refractory", "refractory", "refractory_until"):
        raise SyntaxError(
            f"The name '{identifier}' is used in the refractory mechanism "
            " and should not be used as a variable "
            "name."
        )


Equations.register_identifier_check(check_identifier_refractory)


def add_refractoriness(eqs):
    """
    Extends a given set of equations with the refractory mechanism. New
    parameters are added and differential equations with the "unless refractory"
    flag are changed so that their right-hand side is 0 when the neuron is
    refractory (by multiplication with the ``not_refractory`` variable).

    Parameters
    ----------
    eqs : `Equations`
        The equations without refractory mechanism.

    Returns
    -------
    new_eqs : `Equations`
        New equations, with added parameters and changed differential
        equations having the "unless refractory" flag.
    """
    new_equations = []

    # replace differential equations having the active flag
    for eq in eqs.values():
        if eq.type == DIFFERENTIAL_EQUATION and "unless refractory" in eq.flags:
            # the only case where we have to change anything
            new_code = f"int(not_refractory)*({eq.expr.code})"
            new_equations.append(
                SingleEquation(
                    DIFFERENTIAL_EQUATION,
                    eq.varname,
                    eq.dim,
                    expr=Expression(new_code),
                    flags=eq.flags,
                )
            )
        else:
            new_equations.append(eq)

    # add new parameters
    new_equations.append(
        SingleEquation(PARAMETER, "not_refractory", DIMENSIONLESS, var_type=BOOLEAN)
    )
    new_equations.append(SingleEquation(PARAMETER, "lastspike", second.dim))

    return Equations(new_equations)
