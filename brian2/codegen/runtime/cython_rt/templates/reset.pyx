{% extends 'common.pyx' %}

{% block maincode %}
    {# USES_VARIABLES { _spikespace } #}
    
    # scalar code
    _vectorisation_idx = 1
    {{scalar_code|autoindent}}

    cdef int _num_spikes = {{_spikespace}}[_num{{_spikespace}}-1]
    for _index_spikes in range(_num_spikes):
        # vector code
        _idx = {{_spikespace}}[_index_spikes]
        _vectorisation_idx = _idx
        {{ vector_code | autoindent }}
{% endblock %}
