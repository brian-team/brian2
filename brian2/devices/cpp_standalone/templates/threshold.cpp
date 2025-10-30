{# USES_VARIABLES { N } #}
{% extends 'common_group.cpp' %}
{% block maincode %}
       {# not_refractory and lastspike are added as needed_variables in the
       Thresholder class, we cannot use the USES_VARIABLE mechanism
       conditionally
       Same goes for "eventspace" (e.g. spikespace) which depends on the type of
       event #}

    //// MAIN CODE ////////////
    // scalar code
    const int _vectorisation_idx = -1;
    {{scalar_code|autoindent}}

    {#  Get the name of the array that stores these events (e.g. the spikespace array) #}
    {% set _eventspace = get_array_name(eventspace_variable) %}

    long _count = 0;
    for(size_t _idx=0; _idx<N; _idx++)
    {
        const size_t _vectorisation_idx = _idx;
        {{vector_code|autoindent}}
        if(_cond) {
            {{_eventspace}}[_count++] = _idx;
            {% if _uses_refractory %}
            {{not_refractory}}[_idx] = false;
            {{lastspike}}[_idx] = {{t}};
            {% endif %}
        }
    }
    {{_eventspace}}[N] = _count;
{% endblock %}

{% block after_code %}
{% set _eventspace = get_array_name(eventspace_variable) %}
{{_eventspace}}[N] = 0;  // Note that this is not an off-by-1-error: the array has N+1 elements
{% endblock %}
