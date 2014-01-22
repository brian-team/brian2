{% import 'common_macros.cpp' as common with context %}
{% macro main() %}
    {{ common.insert_group_preamble() }}
    // USES_VARIABLES { _t, _rate, t, dt, _spikespace, _num_source_neurons }
	const int _num_spikes = {{_spikespace}}[_num_spikespace-1];

    // Calculate the new length for the arrays
    const npy_int _new_len = (npy_int)({{_dynamic__t}}.attr("shape")[0]) + 1;

    // Resize the arrays
    PyObject_CallMethod({{_dynamic__t}}, "resize", "i", _new_len);
    PyObject_CallMethod({{_dynamic__rate}}, "resize", "i", _new_len);
    // Get the potentially newly created underlying data arrays
    double *_t_data = (double*)(((PyArrayObject*)(PyObject*){{_dynamic__t}}.attr("data"))->data);
    double *_rate_data = (double*)(((PyArrayObject*)(PyObject*){{_dynamic__rate}}.attr("data"))->data);

    //Set the new values
    _t_data[_new_len - 1] = t;
    _rate_data[_new_len - 1] = 1.0 * _num_spikes / (double)dt / _num_source_neurons;

{% endmacro %}

{% macro support_code() %}
{% endmacro %}
