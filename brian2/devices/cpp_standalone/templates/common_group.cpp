{% macro cpp_file() %}
#include "code_objects/{{codeobj_name}}.h"
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include<cmath>
#include<ctime>
#include<iostream>
#include<fstream>
{% block extra_headers %}
{% endblock %}
{% for name in user_headers %}
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

    {% if openmp_pragma('with_openmp') %}
    const double _start_time = omp_get_wtime();
    {% else %}
    const std::clock_t _start_time = std::clock();
    {% endif %}

	///// CONSTANTS ///////////
	%CONSTANTS%
	///// POINTERS ////////////
	{{pointers_lines|autoindent}}

	{% block maincode %}
	//// MAIN CODE ////////////
	// scalar code
	const int _vectorisation_idx = -1;
	{{scalar_code|autoindent}}

    {# N is a constant in most cases (NeuronGroup, etc.), but a scalar array for
       synapses, we therefore have to take care to get its value in the right
       way. #}
	const int _N = {{constant_or_scalar('N', variables['N'])}};
	{{openmp_pragma('parallel-static')}}
	for(int _idx=0; _idx<_N; _idx++)
	{
	    // vector code
		const int _vectorisation_idx = _idx;
		{% block maincode_inner %}
        {{vector_code|autoindent}}
		{% endblock %}
	}
	{% endblock %}

    {% if openmp_pragma('with_openmp') %}
    const double _run_time = omp_get_wtime() -_start_time;
    {% else %}
    const double _run_time = (double)(std::clock() -_start_time)/CLOCKS_PER_SEC;
    {% endif %}
    {{codeobj_name}}_profiling_info += _run_time;
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
