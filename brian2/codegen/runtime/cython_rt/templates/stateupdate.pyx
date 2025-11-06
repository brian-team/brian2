{# ITERATE_ALL { _idx } #}
{# USES_VARIABLES { N } #}
{# ALLOWS_SCALAR_WRITE #}
{% extends 'common_group.pyx' %}

{% block maincode %}
    # scalar code
    _vectorisation_idx = 1
    {{scalar_code|autoindent}}

    # vector code
    for _idx in range(N):
        _vectorisation_idx = _idx
        {{vector_code|autoindent}}

{% endblock %}
