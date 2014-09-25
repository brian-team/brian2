{% extends 'common.pyx' %}

{% block maincode %}

    {# USES_VARIABLES {_spikespace, N, t, dt, neuron_index, spike_time } #}

    # TODO: We don't deal with more than one spike per neuron yet
    cdef int _cpp_numspikes = 0

    # TODO: improve efficiency, this is O(N) whereas it should be O(1)
    # note that the weave version is O(log N)
    cdef char found = 0
    for _idx in range(_num{{spike_time}}):
        if {{spike_time}}[_idx]>t-dt:
            found = 1
            break
    if not found:
        _idx = _num{{spike_time}}
    
    for _idx in range(_idx, _num{{spike_time}}):
        if {{spike_time}}[_idx]>t:
            break
        {{_spikespace}}[_cpp_numspikes] = {{neuron_index}}[_idx]
        _cpp_numspikes += 1
            
    {{_spikespace}}[N] = _cpp_numspikes

{% endblock %}
