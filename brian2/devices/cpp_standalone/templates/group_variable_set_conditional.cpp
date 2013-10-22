{% extends 'common_group.cpp' %}

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
