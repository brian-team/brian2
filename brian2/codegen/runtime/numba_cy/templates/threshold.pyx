{% extends 'common.pyx' %}
{# USES_VARIABLES { N } #}

{% block maincode %}
    {# t, not_refractory and lastspike are added as needed_variables in the
       Thresholder class, we cannot use the USES_VARIABLE mechanism
       conditionally #}

    # scalar code
    _vectorisation_idx = 1;
    {{ scalar_code | autoindent }}

    cdef long _cpp_numevents = 0

    {#  Get the name of the array that stores these events (e.g. the spikespace array) #}
    {% set _eventspace = get_array_name(eventspace_variable) %}

    for _idx in range(N):
        
        # vector code
        _vectorisation_idx = _idx
        {{ vector_code | autoindent }}

        if _cond:
            {{_eventspace}}[_cpp_numevents] = _idx
            _cpp_numevents += 1
            {% if _uses_refractory %}
            {{not_refractory}}[_idx] = False
            {{lastspike}}[_idx] = {{t}}
            {% endif %}
            
    {{_eventspace}}[N] = _cpp_numevents
    
{% endblock %}
