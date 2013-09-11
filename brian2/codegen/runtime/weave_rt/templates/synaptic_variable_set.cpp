{% extends 'common_group.cpp' %}

{% block maincode %}
	// USES_VARIABLES { _group_idx, _synaptic_pre, _synaptic_post,
    //                  _source_offset, _target_offset }
	//// MAIN CODE ////////////
	for(int _idx_group_idx=0; _idx_group_idx<_num_group_idx; _idx_group_idx++)
	{
		const int _idx = _group_idx[_idx_group_idx];
		const int _presynaptic_idx = _synaptic_pre[_idx] + _source_offset;
		const int _postsynaptic_idx = _synaptic_post[_idx] + _target_offset;
		const int _vectorisation_idx = _idx;
		{{ super() }}
	}
{% endblock %}
