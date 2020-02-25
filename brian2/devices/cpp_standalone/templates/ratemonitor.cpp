{# USES_VARIABLES { N, rate, t, _spikespace, _clock_t, _clock_dt} #}
{# WRITES_TO_READ_ONLY_VARIABLES { N } #}
{% extends 'common_group.cpp' %}

{% block maincode %}
    size_t _num_spikes = {{_spikespace}}[_num_spikespace-1];
    {% if subgroup and not contiguous %}
    // We use the same data structure as for the eventspace to store the
    // "filtered" events, i.e. the events that are indexed in the subgroup
    int32_t _filtered_events[{{source_N}} + 1];
    _filtered_events[{{source_N}}] = 0;
    size_t _source_index_counter = 0;
    {% endif %}
    {% if subgroup %}
    // For subgroups, we do not want to record all spikes
    size_t _start_idx = _num_spikes;
    size_t _end_idx = _num_spikes;
    if (_num_spikes > 0)
    {
        {% if contiguous %}
        for(size_t _j=0; _j<_num_spikes; _j++)
        {
            const int _idx = {{_spikespace}}[_j];
            if (_idx >= _source_start) {
                _start_idx = _j;
                break;
            }
        }
        for(size_t _j=_num_spikes-1; _j>=_start_idx; _j--)
        {
            const int _idx = {{_spikespace}}[_j];
            if (_idx < _source_stop) {
                break;
            }
            _end_idx = _j;
        }
        _num_spikes = _end_idx - _start_idx;
        {% else %}
        for (size_t _j=0; _j<_num_spikes; _j++)
        {
            const size_t _idx = {{_spikespace}}[_j];
            if (_idx < {{_source_indices}}[_source_index_counter])
                continue;
            while ({{_source_indices}}[_source_index_counter] < _idx)
            {
                _source_index_counter++;
            }
            if (_source_index_counter < {{source_N}} &&
                _idx == {{_source_indices}}[_source_index_counter])
            {
                _source_index_counter += 1;
                _filtered_events[_filtered_events[{{source_N}}]++] = _idx;
                if (_source_index_counter == {{source_N}})
                    break;
            }
            if (_source_index_counter == {{source_N}})
                break;
        }
        _num_spikes = _filtered_events[{{source_N}}];
        {% endif %}
    }
    {% endif %}
    {{_dynamic_rate}}.push_back(1.0*_num_spikes/{{_clock_dt}}/{{source_N}});
    {{_dynamic_t}}.push_back({{_clock_t}});
    {{N}}++;
{% endblock %}

