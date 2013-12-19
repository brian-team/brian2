{% extends 'common_group.cpp' %}

{% block maincode %}
    // USES_VARIABLES { _synaptic_post, _num_target_neurons }
	//// MAIN CODE ////////////
	{% if _get_target_var_name is defined %}
	{% set _target_var = _get_target_var_name() %}
	{% endif %}

	// Set all the target variable values to zero
	for (int _target_idx=0; _target_idx<_num_target_neurons; _target_idx++)
	    {{_target_var}}[_target_idx] = 0.0;

	for(int _idx=0; _idx<_num_synaptic_post; _idx++)
	{
	    {{ super() }}
		{{_target_var}}[{{_synaptic_post}}[_idx]] += _synaptic_var;
	}
{% endblock %}
