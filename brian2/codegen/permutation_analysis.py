"""
Module for analysing synaptic pre and post code for synapse order independence.
"""

from brian2.utils.stringtools import get_identifiers
from brian2.core.functions import Function
from brian2.core.variables import Constant

__all__ = ["OrderDependenceError", "check_for_order_independence"]


class OrderDependenceError(Exception):
    pass


def check_for_order_independence(statements, variables, indices):
    """
    Check that the sequence of statements doesn't depend on the order in which the indices are iterated through.
    """
    # Remove stateless functions from variables (only bother with ones that are used)
    all_used_vars = set()
    for statement in statements:
        all_used_vars.update(get_identifiers(statement.expr))
    variables = variables.copy()
    for var in set(variables.keys()).intersection(all_used_vars):
        val = variables[var]
        if isinstance(val, Function):
            if val.stateless:
                del variables[var]
            else:
                raise OrderDependenceError(
                    "Function %s may have internal state, "
                    "which can lead to order dependence." % var
                )
    all_variables = [v for v in variables if not isinstance(variables[v], Constant)]
    # Main index variables are those whose index corresponds to the main index being iterated through. By
    # assumption/definition, these indices are unique, and any order-dependence cannot come from their values,
    # only from the values of the derived indices. In the most common case of Synapses, the main index would be
    # the synapse index, and the derived indices would be pre and postsynaptic indices (which can be repeated).
    unique_index = lambda v: (
        indices[v] != "0" and getattr(variables[indices[v]], "unique", False)
    )
    main_index_variables = {
        v for v in all_variables if indices[v] == "_idx" or unique_index(v)
    }
    different_index_variables = set(all_variables) - main_index_variables

    # At the start, we assume all the different/derived index variables are permutation independent and we continue
    # to scan through the list of statements checking whether or not permutation-dependence has been introduced
    # until the permutation_independent set has stopped changing.
    permutation_independent = list(different_index_variables)
    permutation_dependent_aux_vars = set()
    changed_permutation_independent = True
    for statement in statements:
        if statement.op == ":=" and statement.var not in all_variables:
            main_index_variables.add(statement.var)
            all_variables.append(statement.var)

    while changed_permutation_independent:
        changed_permutation_independent = False
        for statement in statements:
            vars_in_expr = get_identifiers(statement.expr).intersection(all_variables)
            # any time a statement involves a LHS and RHS which only depend on itself, this doesn't change anything
            if {statement.var} == vars_in_expr:
                continue
            nonsyn_vars_in_expr = vars_in_expr.intersection(different_index_variables)
            permdep = any(
                var not in permutation_independent for var in nonsyn_vars_in_expr
            )
            permdep = permdep or any(
                var in permutation_dependent_aux_vars for var in vars_in_expr
            )
            if statement.op == ":=":  # auxiliary variable created
                if permdep:
                    if statement.var not in permutation_dependent_aux_vars:
                        permutation_dependent_aux_vars.add(statement.var)
                        changed_permutation_independent = True
                continue
            elif statement.var in main_index_variables:
                if permdep:
                    raise OrderDependenceError()
            elif statement.var in different_index_variables:
                if statement.op in ("+=", "*=", "-=", "/="):
                    if permdep:
                        raise OrderDependenceError()
                    if statement.var in permutation_independent:
                        permutation_independent.remove(statement.var)
                        changed_permutation_independent = True
                elif statement.op == "=":
                    otheridx = [
                        v
                        for v in variables
                        if indices[v] not in (indices[statement.var], "_idx", "0")
                    ]
                    if any(var in otheridx for var in vars_in_expr):
                        raise OrderDependenceError()
                    if permdep:
                        raise OrderDependenceError()
                    if any(var in main_index_variables for var in vars_in_expr):
                        raise OrderDependenceError()
                else:
                    raise OrderDependenceError()
            else:
                raise AssertionError("Should never get here...")
