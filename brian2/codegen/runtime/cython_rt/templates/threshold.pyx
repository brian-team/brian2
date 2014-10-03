{% extends 'common.pyx' %}
{# USES_VARIABLES { N } #}

{% block maincode %}
    {# USES_VARIABLES {_spikespace } #}
    # t, not_refractory and lastspike are added as needed_variables in the
    # Thresholder class, we cannot use the USES_VARIABLE mechanism
    # conditionally

    # scalar code
    _vectorisation_idx = 1;
    {{ scalar_code | autoindent }}

    cdef long _cpp_numspikes = 0
    
    for _idx in range(N):
        
        # vector code
        _vectorisation_idx = _idx
        {{ vector_code | autoindent }}

        if _cond:
            {{_spikespace}}[_cpp_numspikes] = _idx
            _cpp_numspikes += 1
            {% if _uses_refractory %}
            {{not_refractory}}[_idx] = False
            {{lastspike}}[_idx] = t
            {% endif %}
            
    {{_spikespace}}[N] = _cpp_numspikes
    
{% endblock %}
