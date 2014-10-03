{% extends 'common.pyx' %}

{% block maincode %}

    {# USES_VARIABLES {_spikespace, N, t, dt, neuron_index, spike_time, period, _lastindex } #}

    # TODO: We don't deal with more than one spike per neuron yet
    cdef int _cpp_numspikes = 0
    cdef int padding        = 0
    cdef float epsilon      = 0.001*dt

    if (period > 0):
        padding = int(t / period)

    # If there is a periodicity in the SpikeGenerator, we need to reset the lastindex 
    # when all spikes have been played
    if ((period > 0) and (abs(t-period*padding) < epsilon)):
        {{_lastindex}}[0] = 0      

    for _idx in range({{_lastindex}}[0], _num{{spike_time}}):
        test = ({{spike_time}}[_idx] > (t - period*padding)) or (abs({{spike_time}}[_idx] - (t - period*padding)) < epsilon)
        if test:
            break
        {{_lastindex}}[0] += 1
    
    for _idx in range({{_lastindex}}[0], _num{{spike_time}}):
        test = ({{spike_time}}[_idx] > (t + dt - period*padding)) or (abs({{spike_time}}[_idx] - (t + dt - period*padding)) < epsilon)
        if test:
            break
        {{_spikespace}}[_cpp_numspikes] = {{neuron_index}}[_idx]
        _cpp_numspikes += 1

    {{_spikespace}}[N] = _cpp_numspikes
    {{_lastindex}}[0] += _cpp_numspikes

{% endblock %}
