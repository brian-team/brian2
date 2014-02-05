{% import 'common_macros.cpp' as common with context %}
{% macro main() %}
    {{ common.insert_group_preamble() }}
    {# USES_VARIABLES { t, rate, _clock_t, _clock_dt, _spikespace, _num_source_neurons } #}
	const int _num_spikes = {{_spikespace}}[_num_spikespace-1];

    // Calculate the new length for the arrays
    const npy_int _new_len = (npy_int)({{_dynamic_t}}.attr("shape")[0]) + 1;

    // Resize the arrays
    PyObject_CallMethod(_owner, "resize", "i", _new_len);

    // Get the potentially newly created underlying data arrays
    double *t_data = (double*)(((PyArrayObject*)(PyObject*){{_dynamic_t}}.attr("data"))->data);
    double *rate_data = (double*)(((PyArrayObject*)(PyObject*){{_dynamic_rate}}.attr("data"))->data);

    //Set the new values
    t_data[_new_len - 1] = _clock_t;
    rate_data[_new_len - 1] = 1.0 * _num_spikes / (double)_clock_dt / _num_source_neurons;

{% endmacro %}

{% macro support_code() %}
{% endmacro %}
