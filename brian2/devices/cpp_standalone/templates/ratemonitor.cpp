{# USES_VARIABLES { N, rate, t, _spikespace, _clock_t, _clock_dt,
                    _num_source_neurons, _source_start, _source_stop } #}
{# WRITES_TO_READ_ONLY_VARIABLES { N } #}
{% extends 'common_group.cpp' %}

{% block maincode %}
    size_t _num_spikes = {{_spikespace}}[_num_spikespace-1];
    // For subgroups, we do not want to record all spikes
    // We assume that spikes are ordered
    int _start_idx = -1;
    int _end_idx = -1;
    for(size_t _j=0; _j<_num_spikes; _j++)
    {
        const size_t _idx = {{_spikespace}}[_j];
        if (_idx >= _source_start) {
            _start_idx = _j;
            break;
        }
    }
    if (_start_idx == -1)
        _start_idx = _num_spikes;
    for(size_t _j=_start_idx; _j<_num_spikes; _j++)
    {
        const size_t _idx = {{_spikespace}}[_j];
        if (_idx >= _source_stop) {
            _end_idx = _j;
            break;
        }
    }
    if (_end_idx == -1)
        _end_idx =_num_spikes;
    _num_spikes = _end_idx - _start_idx;
    {{_dynamic_rate}}.push_back(1.0*_num_spikes/{{_clock_dt}}/_num_source_neurons);
    {{_dynamic_t}}.push_back({{_clock_t}});
    {{N}}++;
{% endblock %}
