# USE_SPECIFIERS { _postsynaptic, _presynaptic, _spiking_synapses, _post_neurons }
# TODO: check and improve this
_post_neurons = _postsynaptic.data.take(_spiking_synapses)
_perm = _post_neurons.argsort()
_aux = _post_neurons.take(_perm)
_flag = empty(len(_aux)+1, dtype=bool)
_flag[0] = _flag[-1] = 1
not_equal(_aux[1:], _aux[:-1], _flag[1:-1])
_F = _flag.nonzero()[0][:-1]
logical_not(_flag, _flag)
while len(_F):
    _u = _aux.take(_F)
    _i = _perm.take(_F)
    _postsynaptic_idx = _u
    _synapse_idx = _spiking_synapses[_i]
    # TODO: how do we get presynaptic indices? do we need to?

    {% for line in code_lines %}
    {{line}}
    {% endfor %}

    _F += 1
    _F = extract(_flag.take(_F), _F)
