{# USES_VARIABLES { N, _clock_t, count,
                    _source_start, _source_stop} #}
{% extends 'common_group.pyx' %}

{% block maincode %}

    {#  Get the name of the array that stores these events (e.g. the spikespace array) #}
    {% set _eventspace = get_array_name(eventspace_variable) %}
    cdef size_t _num_events = {{_eventspace}}[_num{{_eventspace}}-1]
    cdef size_t _start_idx, _end_idx, _curlen, _newlen, _j
    {% for varname, var in record_variables | dictsort %}
    cdef {{cpp_dtype(var.dtype)}}* _{{varname}}_ptr
    {% endfor %}
    if _num_events > 0:
        # For subgroups, we do not want to record all spikes
        # We assume that spikes are ordered
        _start_idx = _num_events
        _end_idx = _num_events
        for _j in range(_num_events):
            _idx = {{_eventspace}}[_j]
            if _idx >= _source_start:
                _start_idx = _j
                break
        for _j in range(_num_events-1, _start_idx-1, -1):
            _idx = {{_eventspace}}[_j]
            if _idx < _source_stop:
                break
            _end_idx = _j
        _num_events = _end_idx - _start_idx
        if _num_events > 0:
            # scalar code
            _vectorisation_idx = 1
            {{ scalar_code|autoindent }}
            _curlen = {{N}}
            _newlen = _curlen + _num_events
            # Resize the C++ arrays directly, avoiding Python indirection for efficiency
            {% for varname, var in record_variables | dictsort %}
            {% set dyn_array_name = get_array_name(var, access_data=False) %}
            {{dyn_array_name}}_ptr.resize(_newlen)
            {% endfor %}
            # Update N after resize
            {{N}} = _newlen

            # No we get new fresh pointers after resize
            {% for varname, var in record_variables | dictsort %}
            {% set dyn_array_name = get_array_name(var, access_data=False) %}
            _{{varname}}_ptr = {{dyn_array_name}}_ptr.get_data_ptr()
            {% endfor %}

            # Copy the values across
            for _j in range(_start_idx, _end_idx):
                _idx = {{_eventspace}}[_j]
                _vectorisation_idx = _idx
                {{ vector_code|autoindent }}
                {% for varname in record_variables | sort %}
                _{{varname}}_ptr[_curlen + _j - _start_idx] = _to_record_{{varname}}
                {% endfor %}
                {{count}}[_idx - _source_start] += 1
{% endblock %}
