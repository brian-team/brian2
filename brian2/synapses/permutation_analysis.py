'''
Module for analysing synaptic pre and post code for synapse order independence.
'''
import sympy

from brian2.parsing.sympytools import str_to_sympy, sympy_to_str
from brian2.utils.stringtools import get_identifiers
from brian2.codegen.statements import Statement

__all__ = ['OrderDependenceError', 'check_for_order_independence']


class OrderDependenceError(Exception):
    pass


def check_for_order_independence(statements,
                                 synaptic_variables,
                                 presynaptic_variables, postsynaptic_variables):
    '''
    '''
    # Prepare the statements by using augmented assignments if possible
    # (e.g., replace v_post = v_post + 1*mV by v_post += 1*mV)
    new_statements = []
    for statement in statements:
            sympy_expr = str_to_sympy(statement.expr)
            var = sympy.Symbol(statement.var, real=True)
            collected = sympy.collect(sympy_expr, var, exact=True, evaluate=False)
            if len(collected) == 2 and set(collected.keys()) == {1, var}:
                # We can replace this statement by a += assignment
                new_statements.append(Statement(statement.var,
                                                '+=',
                                                sympy_to_str(collected[1]),
                                                statement.comment,
                                                dtype=statement.dtype))
            elif len(collected) == 1 and var in collected:
                # We can replace this statement by a *= assignment
                new_statements.append(Statement(statement.var,
                                                '*=',
                                                sympy_to_str(collected[var]),
                                                statement.comment,
                                                dtype=statement.dtype))
            else:
                new_statements.append(statement)

    statements = new_statements

    non_synaptic_variables = presynaptic_variables.union(postsynaptic_variables)
    variables = synaptic_variables.union(non_synaptic_variables)
    permutation_independent = non_synaptic_variables.copy()
    changed_permutation_independent = True
    while changed_permutation_independent:
        changed_permutation_independent = False
        for statement in statements:
            vars_in_expr = get_identifiers(statement.expr).intersection(variables)
            nonsyn_vars_in_expr = vars_in_expr.intersection(non_synaptic_variables)
            permdep = any(var not in permutation_independent for var in  nonsyn_vars_in_expr)
            if statement.var in synaptic_variables:
                if permdep:
                    raise OrderDependenceError()
            elif statement.var in non_synaptic_variables:
                if statement.op=='+=' or statement.op=='*=':
                    if permdep:
                        raise OrderDependenceError()
                    if statement.var in permutation_independent:
                        permutation_independent.remove(statement.var)
                        changed_permutation_independent = True
                elif statement.op=='=':
                    if statement.var in presynaptic_variables:
                        sameidx, otheridx = presynaptic_variables, postsynaptic_variables
                    else:
                        sameidx, otheridx = postsynaptic_variables, presynaptic_variables
                    if any(var in otheridx for var in vars_in_expr):
                        raise OrderDependenceError()
                    if permdep:
                        raise OrderDependenceError()
                    if any(var in synaptic_variables for var in vars_in_expr):
                        raise OrderDependenceError()
                else:
                    raise OrderDependenceError()
            else:
                raise OrderDependenceError()


if __name__=='__main__':
    from brian2.codegen.translation import make_statements
    from brian2.core.variables import ArrayVariable
    from brian2 import device
    from numpy import float64
    code = '''
    w_syn = v_pre
    v_pre += 1
    '''
    syn = {'w_syn'}
    presyn = {'u_pre', 'v_pre'}
    postsyn = {'x_post', 'y_post'}
    variables = dict()
    for var in syn.union(presyn).union(postsyn):
        variables[var] = ArrayVariable(var, 1, None, 10, device)
    scalar_statements, vector_statements = make_statements(code, variables, float64)
    check_for_order_independence(vector_statements, syn, presyn, postsyn)
