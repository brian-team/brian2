{% extends 'common_group.cpp' %}

{% block maincode %}
    // USES_VARIABLES { _synaptic_pre, _synaptic_post }
	//// MAIN CODE ////////////
	for(int _idx=0; _idx<_num_idx; _idx++)
	{
	    const int _vectorisation_idx = _idx;
	    const int _presynaptic_idx = _synaptic_pre[_idx];
	    const int _postsynaptic_idx = _synaptic_post[_idx];
	    {{ super() }}
	}
{% endblock %}
