{# USES_VARIABLES { _synaptic_pre, _synaptic_post, sources, targets, N,
 N_incoming, N_outgoing, N_pre, N_post, _source_offset, _target_offset }

#}
{# WRITES_TO_READ_ONLY_VARIABLES { _synaptic_pre, _synaptic_post, N , N_incoming, N_outgoing }
#}
{% extends 'common_group.pyx' %}

{% block maincode %}
    cdef size_t _old_num_synapses = {{N}}
    cdef size_t _new_num_synapses = _old_num_synapses + _num{{sources}}

    # Calculate array sizes
    cdef size_t _N_pre = {{N_pre_val}}
    cdef size_t _N_post = {{N_post_val}}
    cdef int32_t _source_offset_val = {{source_offset_val}}
    cdef int32_t _target_offset_val = {{target_offset_val}}

    # Resize N_incoming/N_outgoing ( they track per-neuron counts)
    {{_dynamic_N_incoming_ptr}}.resize(_N_post + _target_offset_val)
    {{_dynamic_N_outgoing_ptr}}.resize(_N_pre + _source_offset_val)

    # Resize the main synaptic connection arrays
    {{_dynamic__synaptic_pre_ptr}}.resize(_new_num_synapses)
    {{_dynamic__synaptic_post_ptr}}.resize(_new_num_synapses)

    # Get the potentially newly created underlying data arrays
    cdef int32_t* _synaptic_pre_data = {{_dynamic__synaptic_pre_ptr}}.get_data_ptr()
    cdef int32_t* _synaptic_post_data = {{_dynamic__synaptic_post_ptr}}.get_data_ptr()
    cdef int32_t* _N_incoming_data = {{_dynamic_N_incoming_ptr}}.get_data_ptr()
    cdef int32_t* _N_outgoing_data = {{_dynamic_N_outgoing_ptr}}.get_data_ptr()

    for _idx in range(_num{{sources}}):
        # After this code has been executed, the arrays _real_sources and
        # _real_variables contain the final indices. Having any code here it all is
        # only necessary for supporting subgroups
        {{ vector_code | autoindent }}
        _synaptic_pre_data[_idx + _old_num_synapses] = _real_sources
        _synaptic_post_data[_idx + _old_num_synapses] = _real_targets

        # update N_incoming, N_outgoing count
        _N_outgoing_data[_real_sources] += 1
        _N_incoming_data[_real_targets] += 1

    # now we need to resize all registered variables and set the total number
    # of synapses without python indirection
    {% for varname in _registered_variables | variables_to_array_names(access_data=False) | sort %}
    {{varname}}_ptr.resize(_new_num_synapses)
    {% endfor %}
    # Update the total number of synapses
    {{N}} = _new_num_synapses

    # And update N_incoming, N_outgoing and synapse_number
    {% if multisynaptic_index %}
    # Handle multisynaptic index - this requires iteration over all synapses
    # to count how many times each (source, target) pair appears

    cdef dict _source_target_count = {} # Dictionary to track (source, target) pairs
    cdef int32_t _pre_idx , _post_idx
    cdef tuple _pair
    cdef int32_t _count
    cdef int32_t* _synapse_number_data = {{get_array_name(variables[multisynaptic_index], access_data=False)}}_ptr.get_data_ptr()

    for _idx in range(_new_num_synapses):
        _pre_idx = _synaptic_pre_data[_idx]
        _post_idx = _synaptic_post_data[_idx]
        _pair = (_pre_idx, _post_idx)

        _count = _source_target_count.get(_pair,0)
        _synapse_number_data[_idx] = _count
        _source_target_count[_pair] = _count + 1

    {% endif %}
{% endblock %}
