{% extends 'common.pyx' %}

{% block maincode %}
    {# USES_VARIABLES { _spiking_synapses} #}
     # scalar code
    _vectorisation_idx = 1
    {{ scalar_code | autoindent }}

    for _spiking_synapse_idx in range(_num_spiking_synapses):
        # vector code
        _idx = _spiking_synapses[_spiking_synapse_idx]
        _vectorisation_idx = _idx
        {{ vector_code | autoindent }}

{% endblock %}
