{% extends 'common_group.cpp' %}

{% block maincode %}
    {# USES_VARIABLES { _spikespace } #}
    //// MAIN CODE ////////////
    // scalar code
    const int _vectorisation_idx = 1;
    {{scalar_code|autoindent}}

    const int _num_spikes = {{_spikespace}}[_num_spikespace-1];
    for(int _index_spikes=0; _index_spikes<_num_spikes; _index_spikes++)
    {
        // vector code
        const int _idx = {{_spikespace}}[_index_spikes];
        const int _vectorisation_idx = _idx;
        {{ super() }}
    }
{% endblock %}
