{% extends 'common_group.cpp' %}

{% block maincode %}
	//// MAIN CODE ////////////
	for(int _idx=0; _idx<_num_idx; _idx++)
	{
		const int _vectorisation_idx = _idx;
		{{ super() }}
	}
{% endblock %}
