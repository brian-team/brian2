////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{% macro main() %}
	// USE_SPECIFIERS { _num_spikes, _spikes }

	////// SUPPORT CODE ///////
	{% for line in support_code_lines %}
	// {{line}}
	{% endfor %}

	////// HANDLE DENORMALS ///
	{% for line in denormals_code_lines %}
	{{line}}
	{% endfor %}

	////// HASH DEFINES ///////
	{% for line in hashdefine_lines %}
	{{line}}
	{% endfor %}

	///// POINTERS ////////////
	{% for line in pointers_lines %}
	{{line}}
	{% endfor %}

	//// MAIN CODE ////////////
	for(int _index__spikes=0; _index__spikes<_num_spikes; _index__spikes++)
	{
		const int _neuron_idx = _spikes[_index__spikes];
		{% for line in code_lines %}
		{{line}}
		{% endfor %}
	}
{% endmacro %}

////////////////////////////////////////////////////////////////////////////
//// SUPPORT CODE //////////////////////////////////////////////////////////

{% macro support_code() %}
	{% for line in support_code_lines %}
	// {{line}}
	{% endfor %}
{% endmacro %}
