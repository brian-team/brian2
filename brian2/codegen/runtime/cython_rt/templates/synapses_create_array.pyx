{# USES_VARIABLES { _synaptic_pre, _synaptic_post, sources, targets, N,
                  N_incoming, N_outgoing,  N_pre, N_post, _source_offset, _target_offset }
#}
{# WRITES_TO_READ_ONLY_VARIABLES { _synaptic_pre, _synaptic_post, N ,N_incoming, N_outgoing }
#}
{% extends 'common_group.pyx' %}

{% block maincode %}
    cdef size_t _old_num_synapses = {{N}}
    cdef size_t _new_num_synapses = _old_num_synapses + _num{{sources}}
    # Resize the main synaptic connection arrays
    {{_dynamic__synaptic_pre}}.resize(_new_num_synapses)
    {{_dynamic__synaptic_post}}.resize(_new_num_synapses)
    # Get the potentially newly created underlying data arrays
    cdef int32_t* _synaptic_pre_data = {{_dynamic__synaptic_pre_ptr}}.get_data_ptr()
    cdef int32_t* _synaptic_post_data = {{_dynamic__synaptic_post_ptr}}.get_data_ptr()


    for _idx in range(_num{{sources}}):
        # After this code has been executed, the arrays _real_sources and
        # _real_variables contain the final indices. Having any code here it all is
        # only necessary for supporting subgroups
        {{ vector_code | autoindent }}
        _synaptic_pre_data[_idx + _old_num_synapses] = _real_sources
        _synaptic_post_data[_idx + _old_num_synapses] = _real_targets

    # now we need to resize all registered variables and set the total number
    # of synapses without python indirection
    {% for varname in _registered_variables | variables_to_array_names(access_data=False) | sort %}
    {{varname}}.resize(_new_num_synapses)
    {% endfor %}

    # Update the total number of synapses
    {{N}} = _new_num_synapses

    # And update N_incoming, N_outgoing and synapse_number
    _owner._update_synapse_numbers(_old_num_synapses)
{% endblock %}
