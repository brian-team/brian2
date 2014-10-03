{% extends 'common_group.cpp' %}
{% block maincode %}
    {# USES_VARIABLES {_spikespace, N, t, dt, neuron_index, spike_time, period, _lastindex} #}

    int   padding = 0;
    float epsilon = 0.001*dt;

    if (period > 0)
        padding = int(t / period);

    // If there is a periodicity in the SpikeGenerator, we need to reset the lastindex 
    // when all spikes have been played
    if ((period > 0) && (std::abs(t - period*padding) < epsilon))
        {{_lastindex}}[0] = 0;

    // TODO: We don't deal with more than one spike per neuron yet
    long _cpp_numspikes = 0;

    for(int _idx={{_lastindex}}[0]; _idx < _numspike_time; _idx++)
    {
        bool test = ({{spike_time}}[_idx] > (t - period*padding)) || (std::abs({{spike_time}}[_idx] - (t - period*padding)) < epsilon);
        if (test)
            break;
        {{_lastindex}}[0]++;
    }
    
    for(int _idx={{_lastindex}}[0]; _idx < _numspike_time; _idx++)
    {
        bool test = ({{spike_time}}[_idx] > (t + dt - period*padding)) || (std::abs({{spike_time}}[_idx] - (t + dt - period*padding)) < epsilon);
        if (test)
            break;
        {{_spikespace}}[_cpp_numspikes++] = {{neuron_index}}[_idx];
    }       

    {{_spikespace}}[N] = _cpp_numspikes;
    {{_lastindex}}[0] += _cpp_numspikes;
    

{% endblock %}

