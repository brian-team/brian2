{% extends 'common.pyx' %}

{% block maincode %}
    {# USES_VARIABLES { t, i, _clock_t, _spikespace, _count,
                        _source_start, _source_stop} #}
    cdef int _num_spikes = {{_spikespace}}[_num{{_spikespace}}-1]
    cdef int _start_idx, _end_idx, _curlen, _newlen, _j
    cdef double[:] _t_view
    cdef int32_t[:] _i_view
    if _num_spikes > 0:
        # For subgroups, we do not want to record all spikes
        # We assume that spikes are ordered
        # TODO: Will this assumption ever be violated?
        _start_idx = 0
        _end_idx = - 1
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
        if _num_spikes > 0:
            # Get the current length and new length of t and i arrays
            _curlen = {{_dynamic_t}}.shape[0]
            _newlen = _curlen + _num_spikes
            # Resize the arrays
            _owner.resize(_newlen)
            # Get the potentially newly created underlying data arrays
            _t_view = {{_dynamic_t}}.data
            _i_view = {{_dynamic_i}}.data
            # Copy the values across
            for _j in range(_start_idx, _end_idx):
                _idx = {{_spikespace}}[_j]
                _t_view[_curlen + _j - _start_idx] = _clock_t
                _i_view[_curlen + _j - _start_idx] = _idx - _source_start
                {{_count}}[_idx - _source_start] += 1
                
{% endblock %}
