{% extends 'common.pyx' %}

{% block maincode %}
    {# USES_VARIABLES { _spikespace, t, dt, neuron_index, _timebins, _period_bins, _lastindex, timestep, N } #}

    cdef int32_t _the_period    = {{_period_bins}}
    cdef int32_t _timebin       = _timestep({{t}}, {{dt}})
    cdef int32_t _cpp_numspikes = 0;

    if _the_period > 0:
        _timebin %= _the_period
        # If there is a periodicity in the SpikeGenerator, we need to reset the
        # lastindex when the period has passed
        if {{_lastindex}} > 0 and {{_timebins}}[{{_lastindex}} - 1] >= _timebin:
            {{_lastindex}} = 0

    for _idx in range({{_lastindex}}, _num{{_timebins}}):
        if ({{_timebins}}[_idx] > _timebin):
            break

        {{_spikespace}}[_cpp_numspikes] = {{neuron_index}}[_idx]
        _cpp_numspikes += 1

    {{_spikespace}}[N] = _cpp_numspikes

    {{_lastindex}} += _cpp_numspikes

{% endblock %}
