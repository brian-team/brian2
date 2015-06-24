'''
Module for efficient vectorisation of synapses code
'''
from brian2.utils.logger import get_logger
from brian2.utils.stringtools import get_identifiers, word_substitute

__all__ = ['vectorise_synapses_code', 'SynapseVectorisationError']

logger = get_logger(__name__)


class SynapseVectorisationError(Exception):
    pass


def ufunc_at_vectorisation(statements, variables, indices, index):
    '''
    '''
    # We assume that the code has passed the test for synapse order independence

    main_index_variables = [v for v in variables if indices[v] == index]
    
    lines = []
    need_unique_indices = set()
    
    for statement in statements:
        vars_in_expr = get_identifiers(statement.expr).intersection(variables)
        subs = {}
        for var in vars_in_expr:
            subs[var] = '{var}[{idx}]'.format(var=var, idx=indices[var])
        expr = word_substitute(statement.expr, subs)
        if statement.var in main_index_variables:
            line = '{var}[{idx}] {op} {expr}'.format(var=statement.var,
                                                     op=statement.op,
                                                     expr=expr,
                                                     idx=index)
            lines.append(line)
        else:
            if statement.inplace:
                if statement.op=='+=':
                    ufunc_name = '_numpy.add'
                elif statement.op=='*=':
                    ufunc_name = '_numpy.multiply'
                else:
                    raise SynapseVectorisationError()
                line = '{ufunc_name}.at({var}, {idx}, {expr})'.format(ufunc_name=ufunc_name,
                                                                      var=statement.var,
                                                                      idx=indices[statement.var],
                                                                      expr=expr)
                lines.append(line)
            else:
                # if statement is not in-place then we assume the expr has no synaptic
                # variables in it otherwise it would have failed the order independence
                # check. In this case, we only need to work with the unique indices
                need_unique_indices.add(indices[statement.var])
                idx = '_unique_' + indices[statement.var]
                expr = word_substitute(expr, {indices[statement.var]: idx})
                line = '{var}[{idx}] = {expr}'.format(var=statement.var,
                                                      idx=idx, expr=expr)
                lines.append(line)

    for unique_idx in need_unique_indices:
        lines.insert(0, '_unique_{idx} = _numpy.unique({idx})'.format(idx=unique_idx))
        
    return '\n'.join(lines)


def vectorise_synapses_code(statements, variables, indices, index='_idx'):
    try:
        return ufunc_at_vectorisation(statements, variables, indices, index=index)
    except SynapseVectorisationError:
        logger.warn("Failed to vectorise synapses code, falling back on Python loop: note that "
                    "this will be very slow! Switch to another code generation target for "
                    "best performance (e.g. cython or weave).")
        # fall back to loop
        # lines = ['for _idx in xrange(len(_spiking_synapses)):',
        #          '    _syn_idx = _spiking_synapses[_idx]',
        #          '    _pre_idx = _synaptic_pre[_syn_idx]',
        #          '    _post_idx = _synaptic_post[_syn_idx]',
        #          ]
        # non_synaptic_variables = presynaptic_variables.union(postsynaptic_variables)
        # variables = synaptic_variables.union(non_synaptic_variables)
        # subs = {}
        # for var in variables:
        #     if var in synaptic_variables:
        #         idx = '_syn_idx'
        #     elif var in presynaptic_variables:
        #         idx = '_pre_idx'
        #     elif var in postsynaptic_variables:
        #         idx = '_post_idx'
        #     subs[var] = '{var}[{idx}]'.format(var=var, idx=idx)
        # for statement in statements:
        #     line = '    {var} {op} {expr}'.format(var=statement.var, op=statement.op,
        #                                           expr=statement.expr)
        #     line = word_substitute(line, subs)
        #     lines.append(line)
        # return '\n'.join(lines)


if __name__=='__main__':
    from brian2.codegen.translation import make_statements
    from brian2.core.variables import ArrayVariable
    from brian2 import device
    from numpy import float64
    code = '''
    w_syn = v_pre
    v_pre += -a_syn # change operator -= or += to see efficient/inefficient code
    x_post = y_post
    '''
    indices = {'w_syn': '_idx',
               'a_syn': '_idx',
               'u_pre': '_presynaptic_idx',
               'v_pre': '_presynaptic_idx',
               'x_post': '_postsynaptic_idx',
               'y_post': '_postsynaptic_idx'}
    variables = dict()
    for var in indices:
        variables[var] = ArrayVariable(var, 1, None, 10, device)
    scalar_statements, vector_statements = make_statements(code, variables, float64)
    print vectorise_synapses_code(vector_statements, variables, indices)
