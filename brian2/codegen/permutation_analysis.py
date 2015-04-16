'''
Module for analysing synaptic pre and post code for synapse order independence.
'''
from brian2.utils.stringtools import get_identifiers

__all__ = ['OrderDependenceError', 'check_for_order_independence']


class OrderDependenceError(Exception):
    pass


def check_for_order_independence(statements, variables, indices):
    '''
    '''

    main_index_variables = set([v for v in variables
                                if (indices[v] in ('_idx', '0')
                                    or getattr(variables[indices[v]],
                                               'unique',
                                               False))])
    different_index_variables = set(variables.keys()) - main_index_variables
    all_variables = variables.keys()

    permutation_independent = list(different_index_variables)
    changed_permutation_independent = True
    while changed_permutation_independent:
        changed_permutation_independent = False
        for statement in statements:
            vars_in_expr = get_identifiers(statement.expr).intersection(all_variables)
            nonsyn_vars_in_expr = vars_in_expr.intersection(different_index_variables)
            permdep = any(var not in permutation_independent for var in  nonsyn_vars_in_expr)
            if statement.op == ':=':
                continue  # auxiliary variable
            elif statement.var in main_index_variables:
                if permdep:
                    raise OrderDependenceError()
            elif statement.var in different_index_variables:
                if statement.op == '+=' or statement.op == '*=':
                    if permdep:
                        raise OrderDependenceError()
                    if statement.var in permutation_independent:
                        permutation_independent.remove(statement.var)
                        changed_permutation_independent = True
                elif statement.op == '=':
                    sameidx = [v for v in variables
                               if indices[v] == indices[statement.var]]
                    otheridx = [v for v in variables
                                if indices[v] not in (indices[statement.var],
                                                      '_idx', '0')]
                    if any(var in otheridx for var in vars_in_expr):
                        raise OrderDependenceError()
                    if permdep:
                        raise OrderDependenceError()
                    if any(var in main_index_variables for var in vars_in_expr):
                        raise OrderDependenceError()
                else:
                    raise OrderDependenceError()
            else:
                raise AssertionError('Should never get here...')


if __name__=='__main__':
    from brian2.codegen.translation import make_statements
    from brian2.core.variables import ArrayVariable
    from brian2 import device
    from numpy import float64
    code = '''
    w_syn = v_pre
    v_pre += 1
    '''
    indices = {'w_syn': '_idx',
               'u_pre': 'presynaptic_idx',
               'v_pre': 'presynaptic_idx',
               'x_post': 'postsynaptic_idx',
               'y_post': 'postsynaptic_idx'}
    variables = dict()
    for var in indices:
        variables[var] = ArrayVariable(var, 1, None, 10, device)
    scalar_statements, vector_statements = make_statements(code, variables, float64)
    check_for_order_independence(vector_statements, variables, indices)
