{% extends 'common_group.cpp' %}

{% block maincode %}
	// USES_VARIABLES { _spiking_synapses}
	//// MAIN CODE ////////////
	for(int _spiking_synapse_idx=0;
		_spiking_synapse_idx<_num_spiking_synapses;
		_spiking_synapse_idx++)
	{
		const int _idx = _spiking_synapses[_spiking_synapse_idx];
		const int _vectorisation_idx = _idx;
		{% for line in code_lines %}
		{{line}}
		{% endfor %}
	}
{% endblock %}
