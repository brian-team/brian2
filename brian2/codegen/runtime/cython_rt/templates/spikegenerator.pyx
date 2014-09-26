{% extends 'common.pyx' %}

{% block template_support_code %}
cdef int _start_idx = 0
cdef double _current_time = 0.0
{% endblock %}

{% block maincode %}

    {# USES_VARIABLES {_spikespace, N, t, dt, neuron_index, spike_time } #}
     
    global _start_idx, _current_time
    
    # Check if we have been reset, if so reset the cache
    if t<_current_time:
        _start_idx = 0
    
    _current_time = t

    # TODO: We don't deal with more than one spike per neuron yet
    cdef int _cpp_numspikes = 0

    cdef char found = 0
    for _idx in range(_start_idx, _num{{spike_time}}):
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
