{% extends 'common_group.cpp' %}
{# USES_VARIABLES { N } #}

{% block maincode %}
	{# USES_VARIABLES {_spikespace, N, t, dt, neuron_index, spike_time } #}

    // TODO: We don't deal with more than one spike per neuron yet
    long _cpp_numspikes = 0;
	for(int _idx=0; _idx<_numneuron_index; _idx++)
	{
	    const double _spike_time = {{spike_time}}[_idx];
	    if (_spike_time > t-dt)
	    {
            if (_spike_time <= t)
                {{_spikespace}}[_cpp_numspikes++] = {{neuron_index}}[_idx];
            else  // since the spike times are sorted, we can stop here
                break;
        }
	}
	{{_spikespace}}[N] = _cpp_numspikes;
{% endblock %}
