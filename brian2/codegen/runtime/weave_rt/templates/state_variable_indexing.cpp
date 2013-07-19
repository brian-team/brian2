////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{% macro main() %}
	// USE_SPECIFIERS { _num_neurons }
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
	int _cpp_numelements = 0;
	// Container for all the potential indices
	npy_int *_elements = (npy_int *)malloc(sizeof(npy_int) * _num_neurons);
	for(int _neuron_idx=0; _neuron_idx<_num_neurons; _neuron_idx++)
	{
	    const int _vectorisation_idx = _neuron_idx;
		{% for line in code_lines %}
		{{line}}
		{% endfor %}
		if(_cond) {
			_elements[_cpp_numelements++] = _neuron_idx;
		}
	}
	npy_intp _dims[] = {_cpp_numelements};
	PyObject *_numpy_elements = PyArray_SimpleNewFromData(1, _dims, NPY_INT, _elements);
	return_val = _numpy_elements;
{% endmacro %}

////////////////////////////////////////////////////////////////////////////
//// SUPPORT CODE //////////////////////////////////////////////////////////

{% macro support_code() %}
	{% for line in support_code_lines %}
	// {{line}}
	{% endfor %}
{% endmacro %}
