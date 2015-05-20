{% macro cpp_file() %}
#include "code_objects/{{codeobj_name}}.h"
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include<cmath>
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
	///// CONSTANTS ///////////
	%CONSTANTS%
	///// POINTERS ////////////
	{{pointers_lines|autoindent}}

	{% block maincode %}
	//// MAIN CODE ////////////
	// scalar code
	const int _vectorisation_idx = -1;
	{{scalar_code|autoindent}}
	{{openmp_pragma('static')}} 
	for(int _idx=0; _idx<N; _idx++)
	{
	    // vector code
		const int _vectorisation_idx = _idx;
		{% block maincode_inner %}
        {{vector_code|autoindent}}
		{% endblock %}
	}
	{% endblock %}
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
