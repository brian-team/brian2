{% extends 'common_group.cpp' %}
{% block maincode %}
	// USES_VARIABLES { _spikespace }

	{% if variables is defined %}
	{% set _spikespace = variables['_spikespace'].arrayname %}
	{% endif %}

	const int *_spikes = {{_spikespace}};
	const int _num_spikes = {{_spikespace}}[N];

	//// MAIN CODE ////////////
	for(int _index_spikes=0; _index_spikes<_num_spikes; _index_spikes++)
	{
		const int _idx = _spikes[_index_spikes];
		const int _vectorisation_idx = _idx;
		{% for line in code_lines %}
		{{line}}
		{% endfor %}
	}
{% endblock %}
