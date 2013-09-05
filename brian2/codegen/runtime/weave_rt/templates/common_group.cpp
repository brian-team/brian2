{% import 'common_macros.cpp' as common with context %}

{% macro main() %}
	{{ common.insert_group_preamble() }}
	{% block maincode %}{% endblock %}
{% endmacro %}

{% macro support_code() %}
	{{ common.support_code() }}
{% endmacro %}
