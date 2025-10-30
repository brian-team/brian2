{# Note that we use this template only for subexpressions -- for normal arrays
   we do not generate any code but simply access the data in the underlying
   array directly. See RuntimeDevice.get_with_array #}
{# USES_VARIABLES { _group_idx } #}
{% extends 'common_group.pyx' %}

{% block maincode %}

    {%set c_type = cpp_dtype(variables['_variable'].dtype) %}
    {%set np_type = numpy_dtype(variables['_variable'].dtype) %}

    _vectorisation_idx = 1

    cdef size_t _num_elements = 0
    cdef _numpy.ndarray[{{c_type}}, ndim=1, mode='c'] _elements = _numpy.zeros(_num{{_group_idx}}, dtype=_numpy.{{np_type}})
    cdef {{c_type}}[:] _elements_view = _elements

    {{scalar_code|autoindent}}

    for _idx_group_idx in range(_num{{_group_idx}}):
        _idx = {{_group_idx}}[_idx_group_idx]
        _vectorisation_idx = _idx

        {{vector_code|autoindent}}

        _elements_view[_idx_group_idx] = _variable

    return _elements

{% endblock %}
