'''
Module for analysing synaptic pre and post code for synapse order independence.
'''
from brian2.utils.stringtools import get_identifiers, word_substitute, indent, deindent
from numpy.random import randn, randint
from numpy import amax, abs, arange, zeros
from random import shuffle

__all__ = ['OrderDependenceError', 'check_for_order_independence']


class OrderDependenceError(Exception):
    pass


def check_for_order_independence(statements, variables, indices):
    '''
    Check that the sequence of statements doesn't depend on the order in which the indices are iterated through.
    '''
    # Main index variables are those whose index corresponds to the main index being iterated through. By
    # assumption/definition, these indices are unique, and any order-dependence cannot come from their values,
    # only from the values of the derived indices. In the most common case of Synapses, the main index would be
    # the synapse index, and the derived indices would be pre and postsynaptic indices (which can be repeated).
    main_index_variables = set([v for v in variables
                                if (indices[v] in ('_idx', '0')
                                    or getattr(variables[indices[v]],
                                               'unique',
                                               False))])
    different_index_variables = set(variables.keys()) - main_index_variables
    all_variables = variables.keys()
    # At the start, we assume all the different/derived index variables are permutation independent and we continue
    # to scan through the list of statements checking whether or not permutation-dependence has been introduced
    # until the permutation_independent set has stopped changing.
    permutation_independent = list(different_index_variables)
    permutation_dependent_aux_vars = set()
    changed_permutation_independent = True
    for statement in statements:
        if statement.op==':=' and statement.var not in all_variables:
            main_index_variables.add(statement.var)
            all_variables.append(statement.var)
    #index_dependence = dict((k, set([indices[k]])) for k in all_variables)
    while changed_permutation_independent:
        changed_permutation_independent = False
        for statement in statements:
            vars_in_expr = get_identifiers(statement.expr).intersection(all_variables)
            # any time a statement involves a LHS and RHS which only depend on itself, this doesn't change anything
            if set([statement.var])==vars_in_expr:
                continue
            #indices_in_expr = set([indices[k] for k in vars_in_expr])
            nonsyn_vars_in_expr = vars_in_expr.intersection(different_index_variables)
            permdep = any(var not in permutation_independent for var in  nonsyn_vars_in_expr)
            permdep = permdep or any(var in permutation_dependent_aux_vars for var in vars_in_expr)
            if statement.op == ':=': # auxiliary variable created
                if permdep:
                    permutation_dependent_aux_vars.add(statement.var)
                    changed_permutation_independent = True
                continue
            elif statement.var in main_index_variables:
                if permdep:
                    raise OrderDependenceError()
            elif statement.var in different_index_variables:
                if statement.op in ('+=', '*=', '-=', '/='):
                    if permdep:
                        raise OrderDependenceError()
                    if statement.var in permutation_independent:
                        permutation_independent.remove(statement.var)
                        changed_permutation_independent = True
                elif statement.op == '=':
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


# This is an alternative version of the check for order independence that works by numerically trying it out for a
# few values. It needs some work before it could be included.
# def check_for_order_independence(statements, variables, indices):
#     '''
#     Check that the sequence of statements doesn't depend on the order in which the indices are iterated through.
#     '''
#     # Main index variables are those whose index corresponds to the main index being iterated through. By
#     # assumption/definition, these indices are unique, and any order-dependence cannot come from their values,
#     # only from the values of the derived indices. In the most common case of Synapses, the main index would be
#     # the synapse index, and the derived indices would be pre and postsynaptic indices (which can be repeated).
#     main_index_variables = set([v for v in variables
#                                 if (indices[v] in ('_idx', '0')
#                                     or getattr(variables[indices[v]],
#                                                'unique',
#                                                False))])
#     different_index_variables = set(variables.keys()) - main_index_variables
#     all_variables = set(variables.keys())
#     all_variables_plus_aux = all_variables.copy()
#     code = []
#     for statement in statements:
#         vars_in_expr = get_identifiers(statement.expr).intersection(all_variables_plus_aux)
#         op = statement.op
#         if op==':=':
#             op = '='
#             all_variables_plus_aux.add(statement.var)
#         # replace the expression with a sum of the variables used to remove any user functions, note that this will
#         # introduce false positive warnings but it's better than the alternative I think.
#         code.append('{var} {op} {expr}'.format(var=statement.var, op=op,
#                                                expr='+'.join(list(vars_in_expr)+['1'])))
#     code = '\n'.join(code)
#     # numerically checks that a code block used in the test below is permutation-independent by creating a
#     # presynaptic and postsynaptic group of 3 neurons each, and a full connectivity matrix between them, then
#     # repeatedly filling in random values for each of the variables, and checking for several random shuffles of
#     # the synapse order that the result doesn't depend on it. This is a sort of test of the test itself, to make
#     # sure we didn't accidentally assign a good/bad example to the wrong class.
#     code = deindent(code)
#     vals = {}
#     for var in main_index_variables:
#         vals[var] = zeros(10)
#     for var in different_index_variables:
#         vals[var] = zeros(2)
#     subs = dict((var, var+'['+idx+']') for var, idx in indices.iteritems() if var not in indices.values())
#     code = word_substitute(code, subs)
#     index_code = '\n'.join('{idx} = _array{idx}[_idx]'.format(idx=idx) for idx in indices.values() if idx!='_idx')
#     code = '''
# for _idx in _array_idx:
# {index_code}
# {code}
#     '''.format(code=indent(code), index_code=indent(index_code))
#     ns = vals.copy()
#     for _ in xrange(5):
#         origvals = {}
#         for idx in indices.values():
#             if idx=='_idx':
#                 ns['_array'+idx] = arange(10)
#             else:
#                 ns['_array'+idx] = randint(2, size=10)
#         for var in all_variables:
#             v = vals[var]
#             v[:] = randn(len(v))
#             origvals[var] = v.copy()
#         exec code in ns
#         endvals = {}
#         for var in all_variables:
#             endvals[var] = vals[var].copy()
#         for _ in xrange(5):
#             for var in all_variables:
#                 vals[var][:] = origvals[var]
#             shuffle(ns['_array_idx'])
#             exec code in ns
#             for var in all_variables:
#                 if amax(abs(vals[var]-endvals[var]))>1e-5:
#                     raise OrderDependenceError()


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
