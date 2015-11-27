{% extends 'common.pyx' %}

{# USES_VARIABLES { t, _clock_t, _indices, N } #}

{% block maincode %}

    cdef int _new_len = {{N}} + 1

    # Resize the recorded times
    _var_t.resize(_new_len)
    {{_dynamic_t}}[_new_len-1] = {{_clock_t}}

    # scalar code
    _vectorisation_idx = 1
    {{ scalar_code|autoindent }}

    cdef int _i

    {% for varname, var in _recorded_variables.items() %}
    {% set c_type = cpp_dtype(variables[varname].dtype) %}
    {% set np_type = numpy_dtype(variables[varname].dtype) %}
    # Resize the recorded variable "{{varname}}" and get the (potentially
    # changed) reference to the underlying data
    _var_{{varname}}.resize((_new_len, _num{{_indices}}))
    {% if c_type == 'bool'%}
    cdef _numpy.ndarray[char, ndim=2, mode='c', cast=True] _record_buf_{{varname}} = {{get_array_name(var, access_data=False)}}.data
    cdef bool* _record_data_{{varname}} = <{{c_type}}*> _record_buf_{{varname}}.data
    {% else %}
    cdef _numpy.ndarray[{{c_type}}, ndim=2, mode='c'] _record_buf_{{varname}} = {{get_array_name(var, access_data=False)}}.data
    cdef {{c_type}}* _record_data_{{varname}} = <{{c_type}}*> _record_buf_{{varname}}.data
    {% endif %}
    for _i in range(_num{{_indices}}):
        # vector code
        _idx = {{_indices}}[_i]
        _vectorisation_idx = _idx
        
        {{ vector_code | autoindent }}

        _record_data_{{varname}}[(_new_len-1)*_num{{_indices}} + _i] = _to_record_{{varname}}
    {% endfor %}

    # set the N variable explicitly (since we do not call `StateMonitor.resize`)
    {{N}} = _new_len

{% endblock %}
