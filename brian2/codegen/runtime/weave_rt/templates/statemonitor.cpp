{% extends 'common_group.cpp' %}

{% block maincode %}
    // USES_VARIABLES { _t, _clock_t, _indices }
    // Get the current length and new length of t and value arrays
    const int _curlen = _t.attr("shape")[0];
    const int _new_len = _curlen + 1;
    // Resize the arrays
    PyObject_CallMethod(_t, "resize", "i", _new_len);
    {% for _varname in _variable_names %}

    PyObject_CallMethod(_recorded_{{_varname}}, "resize", "((ii))",
                        _new_len, _num_indices);
    {% endfor %}

    // Get the potentially newly created underlying data arrays and copy the
    // data
    double *_t_data = (double*)(((PyArrayObject*)(PyObject*)_t.attr("data"))->data);
    _t_data[_new_len - 1] = _clock_t;

    {% for _varname in _variable_names %}
    {
        PyArrayObject *_record_data = (((PyArrayObject*)(PyObject*)_recorded_{{_varname}}.attr("data")));
        const npy_intp* _record_strides = _record_data->strides;
        for (int _i = 0; _i < _num_indices; _i++)
        {
            const int _idx = _indices[_i];
            const int _vectorisation_idx = _idx;
            {{ super() }}

            // FIXME: This will not work for variables with other data types
            double *recorded_entry = (double*)(_record_data->data + (_new_len - 1)*_record_strides[0] + _i*_record_strides[1]);
            *recorded_entry = _to_record_{{_varname}};
        }
    }
    {% endfor %}

{% endblock %}
