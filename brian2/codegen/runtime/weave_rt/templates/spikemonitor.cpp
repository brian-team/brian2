{% import 'common_macros.cpp' as common with context %}
{% macro main() %}
    {{ common.insert_pointers_lines() }}

    {# USES_VARIABLES { t, i, _clock_t, _spikespace, _count,
                        _source_start, _source_stop} #}
	int _num_spikes = {{_spikespace}}[_num_spikespace-1];
    if (_num_spikes > 0)
    {
        // For subgroups, we do not want to record all spikes
        // We assume that spikes are ordered
        int _start_idx = 0;
        int _end_idx = - 1;
        for(int _j=0; _j<_num_spikes; _j++)
        {
            const int _idx = {{_spikespace}}[_j];
            if (_idx >= _source_start) {
                _start_idx = _j;
                break;
            }
        }
        for(int _j=_start_idx; _j<_num_spikes; _j++)
        {
            const int _idx = {{_spikespace}}[_j];
            if (_idx >= _source_stop) {
                _end_idx = _j;
                break;
            }
        }
        if (_end_idx == -1)
            _end_idx =_num_spikes;
        _num_spikes = _end_idx - _start_idx;
        if (_num_spikes > 0) {
            // Get the current length and new length of t and i arrays
            const int _curlen = {{_dynamic_t}}.attr("shape")[0];
            const int _newlen = _curlen + _num_spikes;
            // Resize the arrays
            py::tuple _newlen_tuple(1);
            _newlen_tuple[0] = _newlen;
            _owner.mcall("resize", _newlen_tuple);
            // Get the potentially newly created underlying data arrays
            double *_t_data = (double*)(((PyArrayObject*)(PyObject*){{_dynamic_t}}.attr("data"))->data);
            // TODO: How to get the correct datatype automatically here?
            npy_int32 *_i_data = (npy_int32*)(((PyArrayObject*)(PyObject*){{_dynamic_i}}.attr("data"))->data);
            // Copy the values across
            for(int _j=_start_idx; _j<_end_idx; _j++)
            {
                const int _idx = {{_spikespace}}[_j];
                _t_data[_curlen + _j - _start_idx] = _clock_t;
                _i_data[_curlen + _j - _start_idx] = _idx - _source_start;
                {{_count}}[_idx - _source_start]++;
            }
        }
	}
{% endmacro %}

{% macro support_code() %}
{% endmacro %}
