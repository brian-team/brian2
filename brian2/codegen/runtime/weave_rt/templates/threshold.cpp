{% extends 'common_group.cpp' %}

{% block maincode %}
	// USES_VARIABLES {_spikespace }
	// t, not_refractory and lastspike are added as needed_variables in the
	// Thresholder class, we cannot use the USES_VARIABLE mechanism
	// conditionally

	//// MAIN CODE ////////////
	long _cpp_numspikes = 0;
	for(int _idx=0; _idx<N; _idx++)
	{
	    const int _vectorisation_idx = _idx;
		{{ super() }}
		if(_cond) {
			_spikespace_array[_cpp_numspikes++] = _idx;
			{% if _uses_refractory %}
			not_refractory_array[_idx] = false;
			lastspike_array[_idx] = t;
			{% endif %}
		}
	}
	_spikespace_array[N] = _cpp_numspikes;
{% endblock %}
