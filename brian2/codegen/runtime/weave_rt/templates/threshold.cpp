////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{% macro main() %}
	// USES_VARIABLES { _num_elements, not_refractory, lastspike, t }
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
	int _cpp_numspikes = 0;
	npy_int32 *_spikes_space = (npy_int32 *)malloc(sizeof(npy_int32) * _num_elements);
	for(int _element_idx=0; _element_idx<_num_elements; _element_idx++)
	{
	    const int _vectorisation_idx = _element_idx;
		{% for line in code_lines %}
		{{line}}
		{% endfor %}
		if(_cond) {
			_spikes_space[_cpp_numspikes++] = _element_idx;
			// We have to use the pointer names directly here: The condition
			// might contain references to not_refractory or lastspike and in
			// that case the names will refer to a single entry.
			_ptr{{_array_not_refractory}}[_element_idx] = false;
			_ptr{{_array_lastspike}}[_element_idx] = t;
		}
	}
	npy_intp _dims[] = {_cpp_numspikes};
	PyObject *_numpy_spikes_array = PyArray_SimpleNewFromData(1, _dims, NPY_INT32, _spikes_space);
	return_val = _numpy_spikes_array;
{% endmacro %}

////////////////////////////////////////////////////////////////////////////
//// SUPPORT CODE //////////////////////////////////////////////////////////

{% macro support_code() %}
	{% for line in support_code_lines %}
	// {{line}}
	{% endfor %}
{% endmacro %}
