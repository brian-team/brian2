////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{% macro cpp_file() %}

#include "code_objects/{{codeobj_name}}.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>

////// SUPPORT CODE ///////
namespace {
	{% for line in support_code_lines %}
	{{line}}
	{% endfor %}
}

////// HASH DEFINES ///////
{% for line in hashdefine_lines %}
{{line}}
{% endfor %}

void _run_{{codeobj_name}}(double t)
{
	///// CONSTANTS ///////////
	%CONSTANTS%
	///// POINTERS ////////////
	{% for line in pointers_lines %}
	{{line}}
	{% endfor %}

	//// MAIN CODE ////////////
	for(int _idx=0; _idx<N; _idx++)
	{
		const int _vectorisation_idx = _idx;
		{% for line in code_lines['condition'] %}
		{{line}}
		{% endfor %}
		if(_cond)
		{
			{% for line in code_lines['statement'] %}
			{{line}}
			{% endfor %}
		}
	}
}
{% endmacro %}

////////////////////////////////////////////////////////////////////////////
//// HEADER FILE ///////////////////////////////////////////////////////////

{% macro h_file() %}
#ifndef _INCLUDED_{{codeobj_name}}
#define _INCLUDED_{{codeobj_name}}

#include "objects.h"

void _run_{{codeobj_name}}(double t);

#endif
{% endmacro %}
