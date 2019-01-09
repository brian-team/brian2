{% extends 'common_group.cpp' %}
{% block maincode %}
    {# USES_VARIABLES { _spikespace, t, dt, neuron_index, _timebins, _period_bins, _lastindex, timestep, N } #}

    const int32_t _the_period    = {{_period_bins}};
    int32_t _timebin             = _timestep({{t}}, {{dt}});
    const int32_t _n_spikes      = 0;

    if (_the_period > 0) {
        _timebin %= _the_period;
        // If there is a periodicity in the SpikeGenerator, we need to reset the
        // lastindex when the period has passed
        if ({{_lastindex}} > 0 && {{_timebins}}[{{_lastindex}} - 1] >= _timebin)
            {{_lastindex}} = 0;
    }

    int32_t _cpp_numspikes = 0;

    for(int _idx={{_lastindex}}; _idx < _num_timebins; _idx++)
    {
        if ({{_timebins}}[_idx] > _timebin)
            break;

        {{_spikespace}}[_cpp_numspikes++] = {{neuron_index}}[_idx];
    }

    {{_spikespace}}[N] = _cpp_numspikes;

    {{_lastindex}} += _cpp_numspikes;

{% endblock %}
