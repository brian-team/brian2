{% extends 'common.pyx' %}

{% block maincode %}

    {# USES_VARIABLES {_spikespace, N, t, dt, neuron_index, spike_time, period, _lastindex } #}

    # TODO: We don't deal with more than one spike per neuron yet
    cdef int _cpp_numspikes   = 0
    cdef float padding_before = t % period
    cdef float padding_after  = (t + dt) % period
    cdef float epsilon        = 1e-3*dt
    cdef double _spike_time

    # We need some precomputed values that will be used during looping
    not_first_spike = {{_lastindex}}[0] > 0
    not_end_period  = abs(padding_after) > (dt - epsilon)

    # If there is a periodicity in the SpikeGenerator, we need to reset the lastindex 
    # when all spikes have been played and at the end of the period
    if not_first_spike and ({{spike_time}}[{{_lastindex}}[0] - 1] > padding_before):
        {{_lastindex}}[0] = 0

    for _idx in range({{_lastindex}}[0], _num{{spike_time}}):
        _spike_time = {{spike_time}}[_idx]
        if not_end_period:
            test = (_spike_time > padding_after) or (abs(_spike_time - padding_after) < epsilon)
        else:
            # If we are in the last timestep before the end of the period, we remove the first part of the
            # test, because padding will be 0
            test = abs(_spike_time - padding_after) < epsilon
        if test:
            break
        {{_spikespace}}[_cpp_numspikes] = {{neuron_index}}[_idx]
        _cpp_numspikes += 1

    {{_spikespace}}[N] = _cpp_numspikes
    {{_lastindex}}[0] += _cpp_numspikes

{% endblock %}
