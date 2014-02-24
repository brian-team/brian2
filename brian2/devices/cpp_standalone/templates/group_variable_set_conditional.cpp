{% extends 'common_group.cpp' %}
{# USES_VARIABLES { N } #}
{% block maincode_inner %}
{{vector_code['condition']|autoindent}}
if(_cond)
{
    {{vector_code['statement']|autoindent}}
}
{% endblock %}
