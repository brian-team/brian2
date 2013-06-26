for i in xrange(_num_source_neurons):
    j = arange(_num_target_neurons)
    
    {% for line in code_lines %}
    {{line}}
    {% endfor %}
    
    _cond_nonzero = _cond.nonzero()[0]
    _cur_num_synapses = len(_presynaptic)
    _numnew = len(_cond_nonzero)
    _new_num_synapses = _cur_num_synapses+_numnew
    _presynaptic.resize(_new_num_synapses)
    _postsynaptic.resize(_new_num_synapses)
    _presynaptic[_cur_num_synapses:] = i
    _postsynaptic[_cur_num_synapses:] = _cond_nonzero
