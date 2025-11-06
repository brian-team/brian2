{# USES_VARIABLES { N, _indices } #}
{% extends 'common_group.pyx' %}

{% block maincode %}

    _vectorisation_idx = 1

    cdef size_t _num_elements = 0
    cdef _numpy.ndarray[int, ndim=1, mode='c'] _elements = _numpy.zeros(N, dtype=_numpy.int32)
    cdef int[:] _elements_view = _elements

    {{scalar_code|autoindent}}

    for _idx in range(N):
        _vectorisation_idx = _idx

        {{vector_code|autoindent}}

        if _cond:
            _elements_view[_num_elements] = _idx
            _num_elements += 1

    return _elements[:_num_elements]

{% endblock %}
