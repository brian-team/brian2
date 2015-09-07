{% extends 'common.pyx' %}

{% block maincode %}
    {# USES_VARIABLES { spiking_synapses} #}
     # scalar code
    _vectorisation_idx = 1
    {{ scalar_code | autoindent }}

    cdef int _spiking_synapse_idx

    for _spiking_synapse_idx in range(_num{{spiking_synapses}}):
        # vector code
        _idx = {{spiking_synapses}}[_spiking_synapse_idx]
        _vectorisation_idx = _idx
        {{ vector_code | autoindent }}

{% endblock %}
