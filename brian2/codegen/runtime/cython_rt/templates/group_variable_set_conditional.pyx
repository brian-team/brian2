{% extends 'common.pyx' %}

{# USES_VARIABLES { N } #}
{# ALLOWS_SCALAR_WRITE #}

{% block maincode %}

    _vectorisation_idx = 1
    
    {{scalar_code['condition']|autoindent}}
    {{scalar_code['statement']|autoindent}}

    for _idx in range(N):
        _vectorisation_idx = _idx
        
        {{vector_code['condition']|autoindent}}
        if _cond:
            {{vector_code['statement']|autoindent}}

{% endblock %}
