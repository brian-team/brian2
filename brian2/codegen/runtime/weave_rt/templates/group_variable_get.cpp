{# Note that we use this template only for subexpressions -- for normal arrays
   we do not generate any code but simply access the data in the underlying
   array directly. See RuntimeDevice.get_with_array #}
{% extends 'common_group.cpp' %}

{% block maincode %}
    {# USES_VARIABLES { _group_idx } #}
    //// MAIN CODE ////////////
    {%set c_type = c_data_type(variables['_variable'].dtype) %}
    {%set numpy_dtype = dtype(variables['_variable'].dtype).char %}
    {%set numpy_type_int = dtype(variables['_variable'].dtype).num %}
    // {{c_type}} {{numpy_dtype}} {{numpy_type_int}}
    int _cpp_numelements = 0;
    // Container for the return values
    {{c_type}}* _elements = ({{c_type}}*)malloc(sizeof({{c_type}}) * _num_group_idx);

    // scalar code
    const int _vectorisation_idx = 1;
    {{scalar_code|autoindent}}
    for(int _idx_group_idx=0; _idx_group_idx<_num_group_idx; _idx_group_idx++)
    {
        // vector code
        const int _idx = {{_group_idx}}[_idx_group_idx];
        const int _vectorisation_idx = _idx;
        {{ super() }}
        _elements[_idx_group_idx] = _variable;
    }
    npy_intp _dims[] = {_num_group_idx};
    PyObject *_numpy_elements = PyArray_SimpleNewFromData(1, _dims, {{numpy_type_int}}, _elements);
    return_val = _numpy_elements;
{% endblock %}
