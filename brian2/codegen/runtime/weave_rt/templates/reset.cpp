{% extends 'common_group.cpp' %}

{% block maincode %}
    //// MAIN CODE ////////////
    // scalar code
    const int _vectorisation_idx = 1;
    {{scalar_code|autoindent}}

    {#  Get the name of the array that stores these events (e.g. the spikespace array) #}
    {% set _eventspace = get_array_name(eventspace_variable) %}
    const int _num_events = {{_eventspace}}[_num{{eventspace_variable.name}}-1];
    for(int _index_events=0; _index_events<_num_events; _index_events++)
    {
        // vector code
        const int _idx = {{_eventspace}}[_index_events];
        const int _vectorisation_idx = _idx;
        {{ super() }}
    }
{% endblock %}
