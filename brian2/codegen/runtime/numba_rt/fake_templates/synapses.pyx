{% extends 'common.pyx' %}

{% block maincode %}
    {# USES_VARIABLES { _queue } #}
    cdef _numpy.ndarray[int32_t, ndim=1, mode='c'] _spiking_synapses = _queue.peek()

    # scalar code
    _vectorisation_idx = 1
    {{ scalar_code | autoindent }}

    cdef int _spiking_synapse_idx

    for _spiking_synapse_idx in range(len(_spiking_synapses)):
        # vector code
        _idx = _spiking_synapses[_spiking_synapse_idx]
        _vectorisation_idx = _idx
        {{ vector_code | autoindent }}

    # Advance the spike queue
    _queue.advance()

{% endblock %}
