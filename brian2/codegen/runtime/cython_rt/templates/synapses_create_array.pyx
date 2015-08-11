{% extends 'common.pyx' %}

{% block maincode %}

    {# USES_VARIABLES { _synaptic_pre, _synaptic_post, sources, targets
                        N_incoming, N_outgoing, N }
    #}
    
    cdef int _old_num_synapses = {{N}}[0]
    cdef int _new_num_synapses = _old_num_synapses + _num{{sources}}

    {{_dynamic__synaptic_pre}}.resize(_new_num_synapses)
    {{_dynamic__synaptic_post}}.resize(_new_num_synapses)
    # Get the potentially newly created underlying data arrays
    cdef int32_t[:] _synaptic_pre_data = {{_dynamic__synaptic_pre}}.data
    cdef int32_t[:] _synaptic_post_data = {{_dynamic__synaptic_post}}.data 
    
    for _idx in range(_num{{sources}}):
        # After this code has been executed, the arrays _real_sources and
        # _real_variables contain the final indices. Having any code here it all is
        # only necessary for supporting subgroups
        {{ vector_code | autoindent }}
        _synaptic_pre_data[_idx + _old_num_synapses] = _real_sources
        _synaptic_post_data[_idx + _old_num_synapses] = _real_targets
        # Update the number of total outgoing/incoming synapses per source/target neuron
        {{N_outgoing}}[_real_sources] += 1
        {{N_incoming}}[_real_targets] += 1
    
    # now we need to resize all registered variables (via Python)
    _owner._resize(_new_num_synapses)
    # Set the total number of synapses
    {{N}}[0] = _new_num_synapses

{% endblock %}
