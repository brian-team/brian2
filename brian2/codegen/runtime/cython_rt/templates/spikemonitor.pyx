{# USES_VARIABLES { N, _clock_t, count} #}
{% extends 'common_group.pyx' %}

{% block maincode %}

    {#  Get the name of the array that stores these events (e.g. the spikespace array) #}
    {% set _eventspace = get_array_name(eventspace_variable) %}

    cdef size_t _num_events = {{_eventspace}}[_num{{_eventspace}}-1]
    cdef size_t _start_idx, _end_idx, _curlen, _newlen, _j
    {% if subgroup and not contiguous %}
    # We use the same data structure as for the eventspace to store the
    # "filtered" events, i.e. the events that are indexed in the subgroup
    cdef int[{{source_N}} + 1] _filtered_events
    cdef size_t _source_index_counter = 0
    _filtered_events[{{source_N}}] = 0
    {% endif %}
    {% for varname, var in record_variables | dictsort %}
    cdef {{cpp_dtype(var.dtype)}}[:] _{{varname}}_view
    {% endfor %}
    if _num_events > 0:
        {% if subgroup %}
        # For subgroups, we do not want to record all spikes
        {% if contiguous %}
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
        {% else %}
        for _j in range(_num_events):
            _idx = {{_eventspace}}[_j]
            if _idx < {{_source_indices}}[_source_index_counter]:
                continue
            if _idx > {{_source_indices}}[{{source_N}}-1]:
                break
            while {{_source_indices}}[_source_index_counter] < _idx:
                _source_index_counter += 1
            if (_source_index_counter < {{source_N}} and
                    _idx == {{_source_indices}}[_source_index_counter]):
                _source_index_counter += 1
                _filtered_events[_filtered_events[{{source_N}}]] = _idx
                _filtered_events[{{source_N}}] += 1
            if _source_index_counter == {{source_N}}:
                break
        _num_events = _filtered_events[{{source_N}}]
        {% endif %}
        {% endif %}
        if _num_events > 0:
            # scalar code
            _vectorisation_idx = 1
            {{ scalar_code|autoindent }}
            _curlen = {{N}}
            _newlen = _curlen + _num_events
            # Resize the arrays
            _owner.resize(_newlen)
            {{N}} = _newlen
            {% for varname, var in record_variables | dictsort %}
            _{{varname}}_view = {{get_array_name(var, access_data=False)}}.data
            {% endfor %}
            # Copy the values across
            {% if subgroup %}
            {% if contiguous %}
            for _j in range(_start_idx, _end_idx):
                _idx = {{_eventspace}}[_j]
                _vectorisation_idx = _idx
                {{ vector_code|autoindent }}
                {% for varname in record_variables | sort %}
                _{{varname}}_view [_curlen + _j - _start_idx] = _to_record_{{varname}}
                {% endfor %}
                {{count}}[_idx - _source_start] += 1
            {% else %}
            for _j in range(_num_events):
                _idx = _filtered_events[_j]
                _vectorisation_idx = _idx
                {{ vector_code|autoindent }}
                {% for varname in record_variables | sort %}
                _{{varname}}_view [_curlen + _j] = _to_record_{{varname}}
                {% endfor %}
                {{count}}[_to_record_i] += 1
            {% endif %}
            {% else %}
            for _j in range(_num_events):
                _idx = {{_eventspace}}[_j]
                _vectorisation_idx = _idx
                {{ vector_code|autoindent }}
                {% for varname in record_variables | sort %}
                _{{varname}}_view [_curlen + _j] = _to_record_{{varname}}
                {% endfor %}
                {{count}}[_idx] += 1
            {% endif %}
{% endblock %}
