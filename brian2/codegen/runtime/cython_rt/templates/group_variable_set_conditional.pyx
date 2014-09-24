{# Note that we use this template only for subexpressions -- for normal arrays
   we do not generate any code but simply access the data in the underlying
   array directly. See RuntimeDevice.get_with_array #}

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
