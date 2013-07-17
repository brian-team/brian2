{% macro main() %}
	// Get the current length and new length of t and i arrays
	int _curlen = t_arr.attr("shape")[0];
	int _newlen = _curlen+_num_spikes;
	// Resize the arrays
	py::tuple _newlen_tuple(1);
	_newlen_tuple[0] = _newlen;
	t_arr.mcall("resize", _newlen_tuple);
	i_arr.mcall("resize", _newlen_tuple);
	// Get the potentially newly created underlying data arrays
	double *_t_arr_data = (double*)(((PyArrayObject*)(PyObject*)t_arr.attr("data"))->data);
	int *_i_arr_data = (int*)(((PyArrayObject*)(PyObject*)i_arr.attr("data"))->data);
	// Copy the values across
	for(int _i=0; _i<_num_spikes; _i++)
	{
		_t_arr_data[_curlen+_i] = t;
		_i_arr_data[_curlen+_i] = _spikes[_i];
	}
{% endmacro %}

{% macro support_code() %}
{% endmacro %}
