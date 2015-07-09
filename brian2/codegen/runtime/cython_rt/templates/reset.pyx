{% extends 'common.pyx' %}

{% block maincode %}
    # scalar code
    _vectorisation_idx = 1
    {{scalar_code|autoindent}}
    {#  Get the name of the array that stores these events (e.g. the spikespace array) #}
    {% set _eventspace = get_array_name(eventspace_variable) %}

    cdef int _num_events = {{_eventspace}}[_num{{_eventspace}}-1]
    for _index_events in range(_num_events):
        # vector code
        _idx = {{_eventspace}}[_index_events]
        _vectorisation_idx = _idx
        {{ vector_code | autoindent }}
{% endblock %}
