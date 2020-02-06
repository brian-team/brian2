{% macro cpp_file() %}
#include "code_objects/{{codeobj_name}}.h"
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include<cmath>
#include<ctime>
#include<iostream>
#include<fstream>
#include<climits>
{% block extra_headers %}
{% endblock %}
{% for name in user_headers | sort %}
#include {{name}}
{% endfor %}

////// SUPPORT CODE ///////
namespace {
    {{support_code_lines|autoindent}}
}

////// HASH DEFINES ///////
{{hashdefine_lines|autoindent}}

void _run_{{codeobj_name}}()
{
    using namespace brian;

    {% if profiled %}
    {% if openmp_pragma('with_openmp') %}
    const double _start_time = omp_get_wtime();
    {% else %}
    const std::clock_t _start_time = std::clock();
    {% endif %}
    {% endif %}

    ///// CONSTANTS ///////////
    %CONSTANTS%
    ///// POINTERS ////////////
    {{pointers_lines|autoindent}}

    {% block maincode %}
    {# Will be overwritten in child templates #}
    {% endblock %}

    {% if profiled %}
    {% if openmp_pragma('with_openmp') %}
    const double _run_time = omp_get_wtime() -_start_time;
    {% else %}
    const double _run_time = (double)(std::clock() -_start_time)/CLOCKS_PER_SEC;
    {% endif %}
    {{codeobj_name}}_profiling_info += _run_time;
    {% endif %}
}

{% block extra_functions_cpp %}
{% endblock %}

{% endmacro %}


{% macro h_file() %}
#ifndef _INCLUDED_{{codeobj_name}}
#define _INCLUDED_{{codeobj_name}}

#include "objects.h"

void _run_{{codeobj_name}}();

{% block extra_functions_h %}
{% endblock %}

#endif
{% endmacro %}
