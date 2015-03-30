{# IS_OPENMP_COMPATIBLE #}
{% extends 'common_group.c' %}

{% block maincode %}
	//// MAIN CODE ////////////
    {# USES_VARIABLES { _clock_t, _spikespace, _count,
                        _source_start, _source_stop} #}
	int32_t _num_spikes = {{_spikespace}}[_num_spikespace-1];
    int _j;

    {{ openmp_pragma('single-nowait') }}
    {
        if (_num_spikes > 0)
        {
            int _start_idx = 0;
            int _end_idx = - 1;
            for(_j=0; _j<_num_spikes; _j++)
            {
                const int _idx = {{_spikespace}}[_j];
                if (_idx >= _source_start) {
                    _start_idx = _j;
                    break;
                }
            }
            for(_j=_start_idx; _j<_num_spikes; _j++)
            {
                const int _idx = {{_spikespace}}[_j];
                if (_idx >= _source_stop) {
                    _end_idx = _j;
                    break;
                }
            }
            if (_end_idx == -1)
                _end_idx =_num_spikes;
            _num_spikes = _end_idx - _start_idx;
            if (_num_spikes > 0) {
                for(_j=_start_idx; _j<_end_idx; _j++)
                {
                    const int _idx = {{_spikespace}}[_j];
                    {{_count}}[_idx-_source_start]++;
                }
            }
        }
    }

{% endblock %}
