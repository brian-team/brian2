{% extends 'common_group.cpp' %}

{% block maincode %}
    // USES_VARIABLES { _t, _clock_t, _indices }

    // Get the current length and new length of t and value arrays
    const int _curlen = {{_object__t}}.attr("shape")[0];
    const int _new_len = _curlen + 1;
    // Resize the arrays
    PyObject_CallMethod({{_object__t}}, "resize", "i", _new_len);
    {% for var in _recorded_variables.values() %}
    PyObject_CallMethod({{get_array_name(var, access_data=False)}}, "resize", "((ii))",
                        _new_len, _num_indices);
    {% endfor %}

    // Get the potentially newly created underlying data arrays and copy the
    // data
    double *_t_data = (double*)(((PyArrayObject*)(PyObject*){{_object__t}}.attr("data"))->data);
    _t_data[_new_len - 1] = _clock_t;

    {% for varname, var in _recorded_variables.items() %}
    {%set c_type = c_data_type(variables[varname].dtype) %}
    {
        PyArrayObject *_record_data = (((PyArrayObject*)(PyObject*){{get_array_name(var, access_data=False)}}.attr("data")));
        const npy_intp* _record_strides = _record_data->strides;
        for (int _i = 0; _i < _num_indices; _i++)
        {
            const int _idx = {{_indices}}[_i];
            const int _vectorisation_idx = _idx;
            {{ super() }}

            {{c_type}} *recorded_entry = ({{c_type}}*)(_record_data->data + (_new_len - 1)*_record_strides[0] + _i*_record_strides[1]);
            *recorded_entry = _to_record_{{varname}};
        }
    }
    {% endfor %}
{% endblock %}
