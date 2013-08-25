{% macro main() %}

    // USES_VARIABLES { _t, _i, t, _spikespace, _count,
    //                  _source_start, _source_end}
	int _num_spikes = _spikespace[_num_spikespace-1];
    if (_num_spikes > 0)
    {
        // For subgroups, we do not want to record all spikes
        // We assume that spikes are ordered
        // TODO: Will this assumption ever be violated?
        int _start_idx = 0;
        int _end_idx = - 1;
        for(int _i=0; _i<_num_spikes; _i++)
        {
            const int _idx = _spikespace[_i];
            if (_idx >= _source_start) {
                _start_idx = _i;
                break;
            }
        }
        for(int _i=_start_idx; _i<_num_spikes; _i++)
        {
            const int _idx = _spikespace[_i];
            if (_idx >= _source_end) {
                _end_idx = _i;
                break;
            }
        }
        if (_end_idx == -1)
            _end_idx =_num_spikes;
        _num_spikes = _end_idx - _start_idx;
        if (_num_spikes > 0) {
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
            npy_int32 *_i_data = (npy_int32*)(((PyArrayObject*)(PyObject*)_i.attr("data"))->data);
            // Copy the values across
            for(int _i=_start_idx; _i<_end_idx; _i++)
            {
                const int _idx = _spikespace[_i];
                _t_data[_curlen + _i - _start_idx] = t;
                _i_data[_curlen + _i - _start_idx] = _idx - _source_start;
                _count[_idx - _source_start]++;
            }
        }
	}
{% endmacro %}

{% macro support_code() %}
{% endmacro %}
