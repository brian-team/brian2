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

    # Get N_pre and N_post (handles both constants and scalar arrays)
    cdef size_t _N_pre = {{constant_or_scalar('N_pre', variables['N_pre'])}}
    cdef size_t _N_post = {{constant_or_scalar('N_post', variables['N_post'])}}

    # Resize N_incoming and N_outgoing statistics arrays
    {{_dynamic_N_incoming}}.resize(_N_post + _target_offset)
    {{_dynamic_N_outgoing}}.resize(_N_pre + _source_offset)

    # Get raw pointers to statistics arrays
    cdef int32_t* _N_incoming_data = {{_dynamic_N_incoming}}.get_data_ptr()
    cdef int32_t* _N_outgoing_data = {{_dynamic_N_outgoing}}.get_data_ptr()

    for _idx in range(_num{{sources}}):
        # After this code has been executed, the arrays _real_sources and
        # _real_variables contain the final indices. Having any code here it all is
        # only necessary for supporting subgroups
        {{ vector_code | autoindent }}
        _synaptic_pre_data[_idx + _old_num_synapses] = _real_sources
        _synaptic_post_data[_idx + _old_num_synapses] = _real_targets

        # Update connection counts
        _N_outgoing_data[_real_sources] += 1
        _N_incoming_data[_real_targets] += 1

    # now we need to resize all registered variables and set the total number
    # of synapses
    {% for varname in owner._registered_variables | variables_to_array_names(access_data=False) | sort %}
    {{varname}}_ptr.resize(newsize)
    {% endfor %}

    # Update the total number of synapses
    {{N}} = newsize

    # And update N_incoming, N_outgoing and synapse_number
    _owner._update_synapse_numbers(_old_num_synapses)
{% endblock %}
