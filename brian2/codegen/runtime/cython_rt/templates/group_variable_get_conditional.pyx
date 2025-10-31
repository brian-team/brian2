{# USES_VARIABLES { N } #}
{% extends 'common_group.pyx' %}

{% block maincode %}

    {%set c_type = cpp_dtype(variables['_variable'].dtype) %}
    {%set np_type = numpy_dtype(variables['_variable'].dtype) %}

    _vectorisation_idx = 1

    cdef size_t _N = {{constant_or_scalar('N', variables['N'])}}
    cdef size_t _num_elements = 0
    cdef _numpy.ndarray[{{c_type}}, ndim=1, mode='c'] _elements = _numpy.zeros(_N, dtype=_numpy.{{np_type}})
    cdef {{c_type}}[:] _elements_view = _elements


    {{scalar_code|autoindent}}

    for _idx in range(_N):
        _vectorisation_idx = _idx

        {{vector_code|autoindent}}

        if _cond:
            _elements_view[_num_elements] = _variable
            _num_elements += 1

    return _elements[:_num_elements]

{% endblock %}
