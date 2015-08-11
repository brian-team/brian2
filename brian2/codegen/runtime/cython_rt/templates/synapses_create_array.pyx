{% extends 'common.pyx' %}

{% block maincode %}

    {# USES_VARIABLES { _synaptic_pre, _synaptic_post, sources, targets
                        N_incoming, N_outgoing, N,
                        N_pre, N_post, _source_offset, _target_offset }
    #}
    
    cdef int _old_num_synapses = {{N}}[0]
    cdef int _new_num_synapses = _old_num_synapses + _num{{sources}}

    {{_dynamic__synaptic_pre}}.resize(_new_num_synapses)
    {{_dynamic__synaptic_post}}.resize(_new_num_synapses)
    # Get the potentially newly created underlying data arrays
    cdef int32_t[:] _synaptic_pre_data = {{_dynamic__synaptic_pre}}.data
    cdef int32_t[:] _synaptic_post_data = {{_dynamic__synaptic_post}}.data 

    # Resize N_incoming and N_outgoing according to the size of the
    # source/target groups
    _var_N_incoming.resize(N_post + _target_offset)
    _var_N_outgoing.resize(N_pre + _source_offset)
    cdef {{cpp_dtype(variables['N_incoming'].dtype)}}[:] _N_incoming = {{_dynamic_N_incoming}}.data.view(_numpy.{{numpy_dtype(variables['N_incoming'].dtype)}})
    cdef {{cpp_dtype(variables['N_outgoing'].dtype)}}[:] _N_outgoing = {{_dynamic_N_outgoing}}.data.view(_numpy.{{numpy_dtype(variables['N_outgoing'].dtype)}})

    for _idx in range(_num{{sources}}):
        # After this code has been executed, the arrays _real_sources and
        # _real_variables contain the final indices. Having any code here it all is
        # only necessary for supporting subgroups
        {{ vector_code | autoindent }}
        _synaptic_pre_data[_idx + _old_num_synapses] = _real_sources
        _synaptic_post_data[_idx + _old_num_synapses] = _real_targets
        # Update the number of total outgoing/incoming synapses per source/target neuron
        _N_outgoing[_real_sources] += 1
        _N_incoming[_real_targets] += 1
    
    # now we need to resize all registered variables (via Python)
    _owner._resize(_new_num_synapses)
    # Set the total number of synapses
    {{N}}[0] = _new_num_synapses

{% endblock %}
