{% extends 'common_group.cpp' %}

{% block maincode %}
    {# USES_VARIABLES { _queue } #}
    //// MAIN CODE ////////////
    // Get the spikes
    const PyArrayObject *_spiking_synapses_obj = (PyArrayObject *)PyObject_CallMethod(_queue, "peek", "");
    const npy_int32 *_spiking_synapses = (npy_int32 *)_spiking_synapses_obj->data;
    const size_t _num_spiking_synapses = _spiking_synapses_obj->dimensions[0];

    // scalar code
    const size_t _vectorisation_idx = 1;
    {{scalar_code|autoindent}}

    for(size_t _spiking_synapse_idx=0;
        _spiking_synapse_idx<_num_spiking_synapses;
        _spiking_synapse_idx++)
    {
        // vector code
        const size_t _idx = _spiking_synapses[_spiking_synapse_idx];
        const size_t _vectorisation_idx = _idx;
        {{vector_code|autoindent}}
    }

    // Advance the spike queue
    PyObject_CallMethod(_queue, "advance", "");
    Py_DECREF(_spiking_synapses_obj);
{% endblock %}
