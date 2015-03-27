'''
Module for analysing synaptic pre and post code for synapse order independence.
'''
from brian2.utils.stringtools import get_identifiers

__all__ = ['OrderDependenceError', 'check_for_order_independence']


class OrderDependenceError(Exception):
    pass


def check_for_order_independence(statements,
                                 synaptic_variables,
                                 presynaptic_variables, postsynaptic_variables):
    '''
    '''
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
    
    