{% extends 'common_group.cpp' %}

{% block maincode %}
	// USES_VARIABLES { _group_idx }
	//// MAIN CODE ////////////
	for(int _idx_group_idx=0; _idx_group_idx<_num_group_idx; _idx_group_idx++)
	{
		const int _idx = _group_idx[_idx_group_idx];
		const int _vectorisation_idx = _idx;
		{% for line in code_lines %}
		{{line}}
		{% endfor %}
	}
{% endblock %}
