{% extends 'common.pyx' %}

{# USES_VARIABLES { t, _clock_t, _indices } #}

{% block maincode %}

    # Get the current length and new length of t and value arrays
    cdef int _curlen = {{_dynamic_t}}.shape[0]
    cdef int _new_len = _curlen + 1

    # Resize the arrays
    _owner.resize(_new_len)
    # Get the potentially newly created underlying data arrays and copy the
    # data (this will translate to a Python statement but it's only one off)
    {{_dynamic_t}}[_new_len-1] = _clock_t

    # scalar code
    _vectorisation_idx = 1
    {{ scalar_code|autoindent }}

    cdef int _i

    {% for varname, var in _recorded_variables.items() %}
    {% set c_type = cpp_dtype(variables[varname].dtype) %}
    {% set np_type = numpy_dtype(variables[varname].dtype) %}
    cdef {{c_type}}[:, :] _record_data_{{varname}} = {{get_array_name(var, access_data=False)}}.data.view(_numpy.{{np_type}})
    for _i in range(_num{{_indices}}):
        # vector code
        _idx = {{_indices}}[_i]
        _vectorisation_idx = _idx
        
        {{ vector_code | autoindent }}

        _record_data_{{varname}}[_new_len-1, _i] = _to_record_{{varname}}
    {% endfor %}

{% endblock %}
