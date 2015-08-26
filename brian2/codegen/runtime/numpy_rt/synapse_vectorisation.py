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
