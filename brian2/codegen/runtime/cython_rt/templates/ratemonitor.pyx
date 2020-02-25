{# USES_VARIABLES { N, t, rate, _clock_t, _clock_dt, _spikespace} #}
{% extends 'common_group.pyx' %}

{% block maincode %}

    cdef size_t _num_spikes = {{_spikespace}}[_num{{_spikespace}}-1]
    {% if subgroup and not contiguous %}
    # We use the same data structure as for the eventspace to store the
    # "filtered" events, i.e. the events that are indexed in the subgroup
    cdef int[{{source_N}} + 1] _filtered_events
    cdef size_t _source_index_counter = 0
    _filtered_events[{{source_N}}] = 0
    {% endif %}
    {% if subgroup %}
    # For subgroups, we do not want to record all spikes
    {% if contiguous %}
    # We assume that spikes are ordered
    _start_idx = _num_spikes
    _end_idx = _num_spikes
    for _j in range(_num_spikes):
        _idx = {{_spikespace}}[_j]
        if _idx >= _source_start:
            _start_idx = _j
            break
    for _j in range(_num_spikes-1, _start_idx-1, -1):
        _idx = {{_spikespace}}[_j]
        if _idx < _source_stop:
            break
        _end_idx = _j
    _num_spikes = _end_idx - _start_idx
    {% else %}
    for _j in range(_num_spikes):
        _idx = {{_spikespace}}[_j]
        if _idx < {{_source_indices}}[_source_index_counter]:
            continue
        while {{_source_indices}}[_source_index_counter] < _idx:
            _source_index_counter += 1
        if (_source_index_counter < {{source_N}} and
                _idx == {{_source_indices}}[_source_index_counter]):
            _source_index_counter += 1
            _filtered_events[_filtered_events[{{source_N}}]] = _idx
            _filtered_events[{{source_N}}] += 1
        if _source_index_counter == {{source_N}}:
            break
    _num_spikes = _filtered_events[{{source_N}}]
    {% endif %}
    {% endif %}
    
    # Calculate the new length for the arrays
    cdef size_t _new_len = {{_dynamic_t}}.shape[0] + 1

    # Resize the arrays
    _owner.resize(_new_len)
    {{N}} = _new_len

    # Set the new values
    {{_dynamic_t}}.data[_new_len-1] = {{_clock_t}}
    {{_dynamic_rate}}.data[_new_len-1] = _num_spikes/{{_clock_dt}}/{{source_N}}

{% endblock %}
