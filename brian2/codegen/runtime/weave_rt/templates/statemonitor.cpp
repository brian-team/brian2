{% extends 'common_group.cpp' %}

{% block maincode %}
    {# USES_VARIABLES { t, _clock_t, _indices, N } #}

    // Get the current length and new length of t and value arrays
    const int _new_len = {{N}} + 1;

    // Resize the recorded times and get the (potentially changed) reference to
    // the underlying data
    PyObject_CallMethod(_var_t, "resize", "i", _new_len);
    double *_t_data = (double*)(((PyArrayObject*)(PyObject*){{_dynamic_t}}.attr("data"))->data);
    _t_data[_new_len - 1] = {{_clock_t}};


    // scalar code
	const int _vectorisation_idx = 1;
	{{scalar_code|autoindent}}

    {% for varname, var in _recorded_variables.items() %}
    {%set c_type = c_data_type(variables[varname].dtype) %}
    {
        // Resize the recorded variable "{{varname}}" and get the (potentially
        // changed) reference to the underlying data
        PyObject_CallMethod(_var_{{varname}}, "resize", "((ii))", _new_len, _num_indices);
        PyArrayObject *_record_data = (((PyArrayObject*)(PyObject*){{get_array_name(var, access_data=False)}}.attr("data")));
        const npy_intp* _record_strides = _record_data->strides;
        for (int _i = 0; _i < _num_indices; _i++)
        {
            // vector code
            const int _idx = {{_indices}}[_i];
            const int _vectorisation_idx = _idx;
            {{ super() }}

            {{c_type}} *recorded_entry = ({{c_type}}*)(_record_data->data + (_new_len - 1)*_record_strides[0] + _i*_record_strides[1]);
            *recorded_entry = _to_record_{{varname}};
        }
    }
    {% endfor %}

    // set the N variable explicitly (since we do not call `StateMonitor.resize`)
    {{N}} = _new_len;
{% endblock %}
