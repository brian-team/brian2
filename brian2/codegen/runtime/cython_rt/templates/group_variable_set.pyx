{# USES_VARIABLES { _group_idx } #}
{# ALLOWS_SCALAR_WRITE #}
{% extends 'common_group.pyx' %}

{% block maincode %}

    cdef size_t _target_idx

    _vectorisation_idx = 1

    {{scalar_code|autoindent}}

    for _idx_group_idx in range(_num{{_group_idx}}):
        _idx = {{_group_idx}}[_idx_group_idx]
        _vectorisation_idx = _idx
        _target_idx = _idx

        {{vector_code|autoindent}}

{% endblock %}
