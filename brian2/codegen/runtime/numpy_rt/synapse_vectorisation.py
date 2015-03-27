'''
Module for efficient vectorisation of synapses code
'''
from brian2.utils.logger import get_logger
from brian2.utils.stringtools import get_identifiers, word_substitute

__all__ = ['vectorise_synapses_code', 'SynapseVectorisationError']

logger = get_logger(__name__)


class SynapseVectorisationError(Exception):
    pass


def ufunc_at_vectorisation(statements,
                           synaptic_variables,
                           presynaptic_variables, postsynaptic_variables):
    '''
    '''
    # We assume that the code has passed the test for synapse order independence
    non_synaptic_variables = presynaptic_variables.union(postsynaptic_variables)
    variables = synaptic_variables.union(non_synaptic_variables)
    
    lines = []
    need_unique_pre = False
    need_unique_post = False
    
    for statement in statements:
        vars_in_expr = get_identifiers(statement.expr).intersection(variables)
        subs = {}
        for var in vars_in_expr:
            if var in synaptic_variables:
                idx = '_spiking_synapses'
            elif var in presynaptic_variables:
                idx = '_pre_neurons'
            elif var in postsynaptic_variables:
                idx = '_post_neurons'
            subs[var] = '{var}[{idx}]'.format(var=var, idx=idx)
        expr = word_substitute(statement.expr, subs)
        if statement.var in synaptic_variables:
            line = '{var}[_spiking_synapses] {op} {expr}'.format(var=statement.var,
                                                                 op=statement.op,
                                                                 expr=expr)
            lines.append(line)
        elif statement.var in non_synaptic_variables:
            if statement.inplace:
                if statement.op=='+=':
                    ufunc_name = '_numpy.add'
                elif statement.op=='*=':
                    ufunc_name = '_numpy.multiply'
                else:
                    raise SynapseVectorisationError()
                if statement.var in presynaptic_variables:
                    idx = '_pre_neurons'
                else:
                    idx = '_post_neurons'
                line = '{ufunc_name}.at({var}, {idx}, {expr})'.format(ufunc_name=ufunc_name,
                                                                     var=statement.var,
                                                                     idx=idx, expr=expr)
                lines.append(line)
            else:
                # if statement is not in-place then we assume the expr has no synaptic
                # variables in it otherwise it would have failed the order independence
                # check. In this case, we only need to work with the unique indices
                if statement.var in presynaptic_variables:
                    need_unique_pre = True
                    idx = '_unique_pre_neurons'
                else:
                    need_unique_post = True
                    idx = '_unique_post_neurons'
                expr = word_substitute(expr, {'_pre_neurons': '_unique_pre_neurons',
                                              '_post_neurons': '_unique_post_neurons'})
                line = '{var}[{idx}] = {expr}'.format(var=statement.var, idx=idx, expr=expr)
                lines.append(line)
        else:
            raise SynapseVectorisationError
        
    if need_unique_pre:
        lines = ['_unique_pre_neurons = _numpy.unique(_pre_neurons)']+lines
    if need_unique_post:
        lines = ['_unique_post_neurons = _numpy.unique(_post_neurons)']+lines
        
    return '\n'.join(lines)


def vectorise_synapses_code(statements,
                            synaptic_variables,
                            presynaptic_variables, postsynaptic_variables):
    try:
        return ufunc_at_vectorisation(statements, synaptic_variables,
                                      presynaptic_variables, postsynaptic_variables)
    except SynapseVectorisationError:
        logger.warn("Failed to vectorise synapses code, falling back on Python loop: note that "
                    "this will be very slow! Switch to another code generation target for "
                    "best performance (e.g. cython or weave).")
        # fall back to loop
        lines = ['for _idx in xrange(len(_spiking_synapses)):',
                 '    _syn_idx = _spiking_synapses[_idx]',
                 '    _pre_idx = _synaptic_pre[_syn_idx]',
                 '    _post_idx = _synaptic_post[_syn_idx]',
                 ]
        non_synaptic_variables = presynaptic_variables.union(postsynaptic_variables)
        variables = synaptic_variables.union(non_synaptic_variables)
        subs = {}
        for var in variables:
            if var in synaptic_variables:
                idx = '_syn_idx'
            elif var in presynaptic_variables:
                idx = '_pre_idx'
            elif var in postsynaptic_variables:
                idx = '_post_idx'
            subs[var] = '{var}[{idx}]'.format(var=var, idx=idx)
        for statement in statements:
            line = '    {var} {op} {expr}'.format(var=statement.var, op=statement.op,
                                                  expr=statement.expr)
            line = word_substitute(line, subs)
            lines.append(line)
        return '\n'.join(lines)


if __name__=='__main__':
    from brian2.codegen.translation import make_statements
    from brian2.core.variables import ArrayVariable
    from brian2 import device
    from numpy import float64
    code = '''
    w_syn = v_pre
    v_pre += a_syn # change operator -= or += to see efficient/inefficient code
    x_post = y_post
    '''
    syn = {'w_syn', 'a_syn'}
    presyn = {'u_pre', 'v_pre'}
    postsyn = {'x_post', 'y_post'}
    variables = dict()
    for var in syn.union(presyn).union(postsyn):
        variables[var] = ArrayVariable(var, 1, None, 10, device)
    scalar_statements, vector_statements = make_statements(code, variables, float64)
    print vectorise_synapses_code(vector_statements, syn, presyn, postsyn)
