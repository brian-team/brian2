{% extends 'common_group.cpp' %}

{% block maincode %}
	//// MAIN CODE ////////////
	int _cpp_numelements = 0;
	// Container for all the potential indices
	npy_int *_elements = (npy_int *)malloc(sizeof(npy_int) * N);
	for(int _idx=0; _idx<N; _idx++)
	{
	    const int _vectorisation_idx = _idx;
	    {{ super() }}
		if(_cond) {
			_elements[_cpp_numelements++] = _idx;
		}
	}
	npy_intp _dims[] = {_cpp_numelements};
	PyObject *_numpy_elements = PyArray_SimpleNewFromData(1, _dims, NPY_INT, _elements);
	return_val = _numpy_elements;
{% endblock %}
