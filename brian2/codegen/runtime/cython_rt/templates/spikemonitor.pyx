{% extends 'common.pyx' %}

{% block maincode %}
    {# USES_VARIABLES { t, i, _clock_t, count,
                        _source_start, _source_stop} #}

    {#  Get the name of the array that stores these events (e.g. the spikespace array) #}
    {% set _eventspace = get_array_name(eventspace_variable) %}

    cdef int _num_events = {{_eventspace}}[_num{{_eventspace}}-1]
    cdef int _start_idx, _end_idx, _curlen, _newlen, _j
    cdef double[:] _t_view
    cdef int32_t[:] _i_view
    {% for varname, var in record_variables.items() %}
    cdef {{cpp_dtype(var.dtype)}}[:] _{{varname}}_view
    {% endfor %}
    if _num_events > 0:
        # For subgroups, we do not want to record all spikes
        # We assume that spikes are ordered
        # TODO: Will this assumption ever be violated?
        _start_idx = _num_events
        _end_idx = _num_events
        for _j in range(_num_events):
            _idx = {{_eventspace}}[_j]
            if _idx >= _source_start:
                _start_idx = _j
                break
        for _j in range(_start_idx, _num_events):
            _idx = {{_eventspace}}[_j]
            if _idx >= _source_stop:
                _end_idx = _j
                break
        _num_events = _end_idx - _start_idx
        if _num_events > 0:
            # scalar code
            _vectorisation_idx = 1
            {{ scalar_code|autoindent }}
            # Get the current length and new length of t and i arrays
            _curlen = {{_dynamic_t}}.shape[0]
            _newlen = _curlen + _num_events
            # Resize the arrays
            _owner.resize(_newlen)
            # Get the potentially newly created underlying data arrays
            _t_view = {{_dynamic_t}}.data
            _i_view = {{_dynamic_i}}.data
            {% for varname, var in record_variables.items() %}
            _{{varname}}_view = {{get_array_name(var, access_data=False)}}.data
            {% endfor %}
            # Copy the values across
            for _j in range(_start_idx, _end_idx):
                _idx = {{_eventspace}}[_j]
                _vectorisation_idx = _idx
                {{ vector_code|autoindent }}
                _t_view[_curlen + _j - _start_idx] = _clock_t
                _i_view[_curlen + _j - _start_idx] = _idx - _source_start
                {% for varname in record_variables %}
                _{{varname}}_view [_curlen + _j - _start_idx] = _to_record_{{varname}}
                {% endfor %}
                {{count}}[_idx - _source_start] += 1
{% endblock %}
