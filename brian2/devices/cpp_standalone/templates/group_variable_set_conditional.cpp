{% extends 'common_group.cpp' %}
{# USES_VARIABLES { N } #}
{% block maincode_inner %}
	{% for line in code_lines['condition'] %}
	{{line}}
	{% endfor %}
	if(_cond)
	{
		{% for line in code_lines['statement'] %}
		{{line}}
		{% endfor %}
	}
{% endblock %}
