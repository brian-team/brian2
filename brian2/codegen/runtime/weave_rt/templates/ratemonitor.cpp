{% macro main() %}

    // USE_SPECIFIERS { _t, _rate, t, dt, _spikes }

    // Calculate the new length for the arrays
    const npy_int _new_len = (npy_int)(_t.attr("shape")[0]) + 1;

    // Resize the arrays
    PyObject_CallMethod(_t, "resize", "i", _new_len);
    PyObject_CallMethod(_rate, "resize", "i", _new_len);
    // Get the potentially newly created underlying data arrays
    double *_t_data = (double*)(((PyArrayObject*)(PyObject*)_t.attr("data"))->data);
    double *_rate_data = (double*)(((PyArrayObject*)(PyObject*)_rate.attr("data"))->data);

    //Set the new values
    _t_data[_new_len - 1] = t;
    _rate_data[_new_len - 1] = 1.0 * _num_spikes / (double)dt / _num_source_neurons;

{% endmacro %}

{% macro support_code() %}
{% endmacro %}
