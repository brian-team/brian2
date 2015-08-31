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
    {# USES_VARIABLES { N } #}
    {# ALLOWS_SCALAR_WRITE #}
	using namespace brian;
	///// CONSTANTS ///////////
	%CONSTANTS%
	///// POINTERS ////////////
	{{pointers_lines|autoindent}}

	{% block maincode %}
	//// MAIN CODE ////////////
	// scalar code
	const int _vectorisation_idx = -1;
	{# Note that the scalar_code['statement'] will not write to any scalar
	   variables (except if the condition is simply 'True' and no vector code
	   is present), it will only read in scalar variables that are used by the
	   vector code. #}
	{{scalar_code['condition']|autoindent}}
	{{scalar_code['statement']|autoindent}}

    {# N is a constant in most cases (NeuronGroup, etc.), but a scalar array for
       synapses, we therefore have to take care to get its value in the right
       way. #}
	const int _N = {{constant_or_scalar('N', variables['N'])}};

	{{ openmp_pragma('parallel-static') }}
	for(int _idx=0; _idx<_N; _idx++)
	{
	    // vector code
		const int _vectorisation_idx = _idx;
		{{vector_code['condition']|autoindent}}
		if (_cond)
		{
            {{vector_code['statement']|autoindent}}
        }
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



