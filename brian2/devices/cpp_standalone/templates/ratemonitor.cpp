{# USES_VARIABLES { N, rate, t, _spikespace, _clock_t, _clock_dt} #}
{# WRITES_TO_READ_ONLY_VARIABLES { N } #}
{% extends 'common_group.cpp' %}

{% block maincode %}
    size_t _num_spikes = {{_spikespace}}[_num_spikespace-1];
    {% if subgroup %}
    int32_t _filtered_spikes = 0;
    size_t _source_index_counter = 0;
    size_t _start_idx = _num_spikes;
    size_t _end_idx = _num_spikes;
    if (_num_spikes > 0)
    {
        {% if contiguous %} {# contiguous subgroup #}
        // We filter the spikes, making use of the fact that they are sorted
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
        _filtered_spikes = _end_idx - _start_idx;
        {% else %} {# non-contiguous subgroup #}
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
                _filtered_spikes += 1;
                if (_source_index_counter == {{source_N}})
                    break;
            }
            if (_source_index_counter == {{source_N}})
                break;
        }
        {% endif %}
        _num_spikes = _filtered_spikes;
    }
    {% endif %}
    {{_dynamic_rate}}.push_back(1.0*_num_spikes/{{_clock_dt}}/{{source_N}});
    {{_dynamic_t}}.push_back({{_clock_t}});
    {{N}}++;
{% endblock %}

