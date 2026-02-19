{# USES_VARIABLES { N, count, _clock_t, _source_start, _source_stop, _source_N } #}
{# WRITES_TO_READ_ONLY_VARIABLES { N, count } #}
{% extends 'common_group.cpp' %}

{% block maincode %}
    {# Get the spikespace array name #}
    {% set _eventspace = get_array_name(eventspace_variable) %}

    int32_t _num_events = {{ _eventspace }}[_num{{ eventspace_variable.name }} - 1];

    if (_num_events > 0) {
        // ── Filter for subgroup range ──
        size_t _start_idx = _num_events;
        size_t _end_idx = _num_events;

        for (size_t _j = 0; _j < (size_t)_num_events; _j++) {
            const int _idx = {{ _eventspace }}[_j];
            if (_idx >= _source_start) {
                _start_idx = _j;
                break;
            }
        }
        for (size_t _j = _num_events - 1; _j >= _start_idx; _j--) {
            const int _idx = {{ _eventspace }}[_j];
            if (_idx < _source_stop) {
                break;
            }
            _end_idx = _j;
        }
        _num_events = _end_idx - _start_idx;

        if (_num_events > 0) {
            // Scalar code
            const size_t _vectorisation_idx = 1;
            {{ scalar_code | autoindent }}

            size_t _curlen = {{ N }};
            size_t _newlen = _curlen + _num_events;

            // ── Resize all recorded dynamic arrays via capsules ──
            {% for varname, var in record_variables | dictsort %}
            {% set _dyn_name = get_array_name(var, access_data=False) %}
            {% set _capsule_name = _dyn_name + "_capsule" %}
            {% set _rec_ctype = c_data_type(var.dtype) %}
            {
                auto* _dyn_{{ varname }} = _extract_dynamic_array_1d<{{ _rec_ctype }}>({{ _capsule_name }});
                _dyn_{{ varname }}->resize(_newlen);
            }
            {% endfor %}

            // Update N after resize
            {{ N }} = _newlen;

            // ── Record each spike ──
            for (size_t _j = _start_idx; _j < _end_idx; _j++) {
                const size_t _idx = {{ _eventspace }}[_j];
                const size_t _vectorisation_idx = _idx;
                {{ vector_code | autoindent }}

                {% for varname, var in record_variables | dictsort %}
                {% set _dyn_name = get_array_name(var, access_data=False) %}
                {% set _rec_ctype = c_data_type(var.dtype) %}
                {
                    auto* _dyn = _extract_dynamic_array_1d<{{ _rec_ctype }}>({{ _dyn_name }}_capsule);
                    _dyn->get_data_ptr()[_curlen + _j - _start_idx] = _to_record_{{ varname }};
                }
                {% endfor %}

                {{ count }}[_idx - _source_start]++;
            }
        }
    }
{% endblock %}
