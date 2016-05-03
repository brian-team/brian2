{% extends 'common.pyx' %}

{# USES_VARIABLES { _group_idx } #}
{# ALLOWS_SCALAR_WRITE #}

{% block maincode %}

    cdef int _target_idx
    
    _vectorisation_idx = 1
    
    {{scalar_code|autoindent}}
    
    for _idx_group_idx in range(_num{{_group_idx}}):
        _idx = {{_group_idx}}[_idx_group_idx]
        _vectorisation_idx = _idx
        _target_idx = _idx
        
        {{vector_code|autoindent}}

{% endblock %}
