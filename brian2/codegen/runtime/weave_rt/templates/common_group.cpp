{% import 'common_macros.cpp' as common with context %}

{% macro main() %}
	{{ common.insert_group_preamble() }}
	{% block maincode %}
    {{vector_code|autoindent}}
	{% endblock %}
{% endmacro %}



{% macro support_code() %}
{% block support_code_block %}
	{{ common.support_code() }}
{% endblock %}
{% endmacro %}
