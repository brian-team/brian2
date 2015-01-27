{% extends 'common_group.cpp' %}
{% block maincode %}
    {# USES_VARIABLES {_spikespace, N, t, dt, neuron_index, spike_time, period, _lastindex} #}

    double padding_before = fmod(t, period);
    double padding_after  = fmod(t+dt, period);
    double epsilon        = 1e-3*dt;

    // We need some precomputed values that will be used during looping
    bool not_first_spike = ({{_lastindex}}[0] > 0);
    bool not_end_period  = (fabs(padding_after) > epsilon);
    bool test;

    // TODO: We don't deal with more than one spike per neuron yet
    long _cpp_numspikes = 0;

    // If there is a periodicity in the SpikeGenerator, we need to reset the lastindex 
    // when all spikes have been played and at the end of the period
    if (not_first_spike && ({{spike_time}}[{{_lastindex}}[0] - 1] > padding_before))
    {   
        {{_lastindex}}[0] = 0;
    }

    for(int _idx={{_lastindex}}[0]; _idx < _numspike_time; _idx++)
    {
        if (not_end_period)
            test = ({{spike_time}}[_idx] > padding_after) || (fabs({{spike_time}}[_idx] - padding_after) < epsilon);
        else
            // If we are in the last timestep before the end of the period, we remove the first part of the
            // test, because padding will be 0
            test = (fabs({{spike_time}}[_idx] - padding_after) < epsilon);
        if (test)
            break;
        {{_spikespace}}[_cpp_numspikes++] = {{neuron_index}}[_idx];
    }       

    {{_spikespace}}[N] = _cpp_numspikes;
    {{_lastindex}}[0] += _cpp_numspikes;
    

{% endblock %}


