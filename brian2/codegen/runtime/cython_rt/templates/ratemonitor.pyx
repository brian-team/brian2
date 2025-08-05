{# USES_VARIABLES { N, t, rate, _clock_t, _clock_dt, _spikespace,
                    _num_source_neurons, _source_start, _source_stop } #}
{% extends 'common_group.pyx' %}

{% block maincode %}

    cdef size_t _num_spikes = {{_spikespace}}[_num{{_spikespace}}-1]

    # For subgroups, we do not want to record all spikes
    # We assume that spikes are ordered
    cdef int _start_idx = -1
    cdef int _end_idx = -1
    cdef size_t _j
    for _j in range(_num_spikes):
        _idx = {{_spikespace}}[_j]
        if _idx >= _source_start:
            _start_idx = _j
            break
    if _start_idx == -1:
        _start_idx = _num_spikes
    for _j in range(_start_idx, _num_spikes):
        _idx = {{_spikespace}}[_j]
        if _idx >= _source_stop:
            _end_idx = _j
            break
    if _end_idx == -1:
        _end_idx =_num_spikes
    _num_spikes = _end_idx - _start_idx

    # First we get the current size of array from the C++ Object itself
    {% set t_array = get_array_name(variables['t'],access_data=False) %}
    {% set rate_array = get_array_name(variables['rate'],access_data=False) %}
    cdef size_t _current_len = {{t_array}}_ptr.size()
    cdef size_t _new_len = _current_len + 1

    # Now we resize the arrays directly , avoiding python indirection
    {{t_array}}_ptr.resize(_new_len)
    {{rate_array}}_ptr.resize(_new_len)

    # Update N after resizing
    {{N}} = _new_len

    cdef double* _t_data = {{t_array}}_ptr.get_data_ptr()
    cdef double* _rate_data = {{rate_array}}_ptr.get_data_ptr()

    # At last we set the new values using the new pointers
    _t_data[_new_len-1] = {{_clock_t}}
    _rate_data[_new_len-1] = _num_spikes/{{_clock_dt}}/_num_source_neurons

{% endblock %}
