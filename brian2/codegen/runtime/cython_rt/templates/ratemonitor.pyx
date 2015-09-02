{% extends 'common.pyx' %}

{% block maincode %}
    {# USES_VARIABLES { t, rate, _clock_t, _clock_dt, _spikespace,
                        _num_source_neurons, _source_start, _source_stop } #}

    cdef int _num_spikes = {{_spikespace}}[_num{{_spikespace}}-1]
    
    # For subgroups, we do not want to record all spikes
    # We assume that spikes are ordered
    cdef int _start_idx = 0
    cdef int _end_idx = -1
    cdef int _j
    for _j in range(_num_spikes):
        _idx = {{_spikespace}}[_j]
        if _idx >= _source_start:
            _start_idx = _j
            break
    for _j in range(_start_idx, _num_spikes):
        _idx = {{_spikespace}}[_j]
        if _idx >= _source_stop:
            _end_idx = _j
            break
    if _end_idx == -1:
        _end_idx =_num_spikes
    _num_spikes = _end_idx - _start_idx
    
    # Calculate the new length for the arrays
    cdef int _new_len = {{_dynamic_t}}.shape[0] + 1

    # Resize the arrays
    _owner.resize(_new_len)

    # Set the new values
    {{_dynamic_t}}.data[_new_len-1] = {{_clock_t}}
    {{_dynamic_rate}}.data[_new_len-1] = _num_spikes/{{_clock_dt}}/_num_source_neurons

{% endblock %}
