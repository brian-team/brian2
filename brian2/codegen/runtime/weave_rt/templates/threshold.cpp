{% extends 'common_group.cpp' %}

{% block maincode %}
	// USES_VARIABLES { not_refractory, lastspike, t, _spikespace }
	//// MAIN CODE ////////////
	long _cpp_numspikes = 0;
	for(int _idx=0; _idx<_num_idx; _idx++)
	{
	    const int _vectorisation_idx = _idx;
		{{ super() }}
		if(_cond) {
			_spikespace[_cpp_numspikes++] = _idx;
			// We have to use the pointer names directly here: The condition
			// might contain references to not_refractory or lastspike and in
			// that case the names will refer to a single entry.
			_ptr{{_array_not_refractory}}[_idx] = false;
			_ptr{{_array_lastspike}}[_idx] = t;
		}
	}
	_spikespace[_num_idx] = _cpp_numspikes;
{% endblock %}
