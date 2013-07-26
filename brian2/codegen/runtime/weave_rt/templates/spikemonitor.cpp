{% macro main() %}

    // USE_SPECIFIERS { _t, _i, t, _spikes, _count }

    if (_num_spikes > 0)
    {
        // Get the current length and new length of t and i arrays
        const int _curlen = _t.attr("shape")[0];
        const int _newlen = _curlen + _num_spikes;
        // Resize the arrays
        py::tuple _newlen_tuple(1);
        _newlen_tuple[0] = _newlen;
        _t.mcall("resize", _newlen_tuple);
        _i.mcall("resize", _newlen_tuple);
        // Get the potentially newly created underlying data arrays
        double *_t_data = (double*)(((PyArrayObject*)(PyObject*)_t.attr("data"))->data);
        // TODO: How to get the correct datatype automatically here?
        npy_int64 *_i_data = (npy_int64*)(((PyArrayObject*)(PyObject*)_i.attr("data"))->data);
        // Copy the values across
        for(int _idx=0; _idx<_num_spikes; _idx++)
        {
            const int _element_idx = _spikes[_idx];
            _t_data[_curlen + _idx] = t;
            _i_data[_curlen + _idx] = _element_idx;
            _count[_element_idx]++;
        }
	}
{% endmacro %}

{% macro support_code() %}
{% endmacro %}
