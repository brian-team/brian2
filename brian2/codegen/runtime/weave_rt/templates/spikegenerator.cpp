{% extends 'common_group.cpp' %}
{% block maincode %}
	{# USES_VARIABLES {_spikespace, N, t, dt, neuron_index, spike_time } #}

    // TODO: We don't deal with more than one spike per neuron yet
    long _cpp_numspikes = 0;
    // Note that std:upper_bound returns a pointer but we want indices
    const unsigned int _start_idx = std::upper_bound({{spike_time}},
                                                     {{spike_time}} + _numspike_time,
                                                     t-dt) - {{spike_time}};
    const unsigned int _stop_idx = std::upper_bound({{spike_time}},
                                                    {{spike_time}} + _numspike_time,
                                                    t) - {{spike_time}};

	for(int _idx=_start_idx; _idx<_stop_idx; _idx++)
	{
        {{_spikespace}}[_cpp_numspikes++] = {{neuron_index}}[_idx];
	}
	{{_spikespace}}[N] = _cpp_numspikes;
{% endblock %}
