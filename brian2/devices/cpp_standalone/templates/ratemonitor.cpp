{% extends 'common_group.cpp' %}

{% block maincode %}
	{# USES_VARIABLES { rate, t, _spikespace, _clock_t, _clock_dt, _num_source_neurons } #}

	int _num_spikes = {{_spikespace}}[_num_spikespace-1];
	{{_dynamic_rate}}.push_back(1.0*_num_spikes/_clock_dt/_num_source_neurons);
	{{_dynamic_t}}.push_back(_clock_t);
{% endblock %}

