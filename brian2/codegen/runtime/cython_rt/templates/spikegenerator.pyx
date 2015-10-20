{% extends 'common.pyx' %}

{% block maincode %}

    {# USES_VARIABLES {_spikespace, N, t, dt, neuron_index, spike_time, period, _lastindex } #}

    # TODO: We don't deal with more than one spike per neuron yet
    cdef int _cpp_numspikes   = 0
    cdef double _the_period = {{period}}
    cdef double padding_before = {{t}} % _the_period
    cdef double padding_after  = ({{t}} + {{dt}}) % _the_period
    cdef double epsilon        = 1e-3*{{dt}}
    cdef double _spike_time

    # We need some precomputed values that will be used during looping
    cdef bool not_end_period  = abs(padding_after) > ( {{dt}} - epsilon) and abs(padding_after) < (_the_period - epsilon)
    cdef bool test

    for _idx in range({{_lastindex}}, _num{{spike_time}}):
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

    # If there is a periodicity in the SpikeGenerator, we need to reset the lastindex
    # when all spikes have been played and at the end of the period
    if not_end_period:
        {{_lastindex}} += _cpp_numspikes
    else:
        {{_lastindex}} = 0

{% endblock %}
